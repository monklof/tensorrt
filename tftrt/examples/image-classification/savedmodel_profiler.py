# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import os
import json
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile
from preprocessing import  vgg_preprocessing as vgg

import time
import numpy as np


class LoggerHook(tf.train.SessionRunHook):
    """Logs runtime of each iteration"""

    def __init__(self, batch_size, num_records, display_every,
                 num_warmup_iterations,
                 enable_profile=False,
                 profile_sample_steps=10,
                 profile_threshold_ms=10.0,
                 profile_output_dir="profile_data",
                 profile_show_dataflow=True,
                 profile_show_memory=False):
        self.iter_times = []
        self.display_every = display_every
        self.num_steps = (num_records + batch_size - 1) / batch_size
        self.batch_size = batch_size
        self.num_warmup_iterations = num_warmup_iterations
        self.enable_profile = enable_profile
        self._output_file = os.path.join(profile_output_dir, "timeline-%d-%.2fms.json")
        self._file_writer = SummaryWriterCache.get(profile_output_dir)
        self._show_dataflow = profile_show_dataflow
        self._show_memory = profile_show_memory
        self._threshold_secs = profile_threshold_ms / 1000.0
        self._current_step = 0
        self._sample_steps = profile_sample_steps

    def begin(self):
        self.start_time = time.time()

    def before_run(self, run_context):
        self._request_summary = (
                self.enable_profile and
                self._current_step >= self.num_warmup_iterations
                and not ((self._current_step + 1) % self._sample_steps))
        opts = (config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
                if self._request_summary else None)

        return SessionRunArgs({}, options=opts)

    def after_run(self, run_context, run_values):
        current_time = time.time()
        duration = current_time - self.start_time
        self.start_time = current_time
        self.iter_times.append(duration)
        self._current_step = len(self.iter_times)
        if self._current_step % self.display_every == 0:
            last_loop_time = sum(self.iter_times[self._current_step - self.display_every:])

            print("    step %d/%d, iter_time(ms)=%.4f, images/sec=%.2f;\t"
                  "last %d: iter_time(s)=%.4f, images/sec=%.2f" % (
                      self._current_step, self.num_steps, duration * 1000,
                      self.batch_size / self.iter_times[-1],
                      self.display_every, last_loop_time,
                      self.batch_size * self.display_every / last_loop_time))

        if self.enable_profile and self._request_summary and \
                self.iter_times[-1] >= self._threshold_secs:
            self._save(self._current_step,
                       self._output_file % (self._current_step,
                                            self.iter_times[-1] * 1000),
                       run_values.run_metadata.step_stats)
            self._file_writer.add_run_metadata(run_values.run_metadata,
                                               "step_%d" % self._current_step)

    def _save(self, step, save_path, step_stats):
        with gfile.Open(save_path, "w") as f:
            trace = timeline.Timeline(step_stats)
            f.write(
                trace.generate_chrome_trace_format(
                    show_dataflow=self._show_dataflow,
                    show_memory=self._show_memory))


def get_image_content(decode=False):
    IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
    if not os.path.exists('cat.jpg'):
        os.system('wget %s' % IMAGE_URL)
    with open('cat.jpg', 'rb') as f:
        content = f.read()

    if decode:
        with tf.device('/cpu:0'):
            with tf.Session(graph=tf.Graph()) as sess:
                image = preprocess_image(content)
                results = sess.run([image])
        return results[0]
    else:
        return content

IMAGE_SIZE = 224

def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  image = tf.image.decode_jpeg(image_buffer, channels=3)
  # image = vgg._aspect_preserving_resize(image, vgg._RESIZE_SIDE_MAX)
  image = vgg._aspect_preserving_resize(image, vgg._RESIZE_SIDE_MIN)
  image = vgg._central_crop([image], IMAGE_SIZE, IMAGE_SIZE)[0]
  image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
  image = tf.to_float(image)
  image = vgg._mean_image_subtraction(image, [vgg._R_MEAN, vgg._G_MEAN, vgg._B_MEAN])
  return image

def run(saved_model_dir, signature_def_key, batch_size,
        num_iterations, num_warmup_iterations,
        display_every=100, enable_profile=False,
        only_run_backbone=False,
        print_results=False,
        profile_output_dir="trace_log", profile_threshold_ms=30,
        profile_sample_steps=10):
    """Evaluates a frozen graph

    This function evaluates a graph on the ImageNet validation set.
    tf.estimator.Estimator is used to evaluate the accuracy of the model
    and a few other metrics. The results are returned as a dict.

    frozen_graph: GraphDef, a graph containing input node 'input' and outputs 'logits' and 'classes'
    model: string, the model name (see NETS table in graph.py)
    data_files: List of TFRecord files used for inference
    batch_size: int, batch size for TensorRT optimizations
    num_iterations: int, number of iterations(batches) to run for
    """

    # Evaluate model
    logger_hook = LoggerHook(
        display_every=display_every,
        batch_size=batch_size,
        num_records=1000,
        num_warmup_iterations=num_warmup_iterations,
        enable_profile=enable_profile,
        profile_output_dir=profile_output_dir,
        profile_threshold_ms=profile_threshold_ms,
        profile_sample_steps=profile_sample_steps)
    hooks = [logger_hook]
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    tag_set = tf.saved_model.tag_constants.SERVING
    meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir,
                                                          tag_set)

    # Construct input feed dict
    # input_tensor_infos = meta_graph_def.signature_def[signature_def_key].inputs
    # input_tensor_names = [input_tensor_infos[tensor_key].name for tensor_key in input_tensor_infos]
    # input_feed_dict = {name: [data] for name in input_tensor_names}
    # Get outputs
    output_tensor_infos = meta_graph_def.signature_def[signature_def_key].outputs
    output_tensor_names = [output_tensor_infos[tensor_key].name for tensor_key in output_tensor_infos]

    graph = tf.Graph()
    with graph.as_default():
        with tf.train.SingularMonitoredSession(config=tf_config,
                                               hooks=hooks) as sess:
            sess.graph._unsafe_unfinalize()
            input_map = None
            if only_run_backbone:
                # image_placeholder = tf.placeholder(tf.float32, shape=(1, 224, 224, 3),
                #                                    name='ImagesPlaceHolder')
                # feed_tensor_name = 'map/TensorArrayStack/TensorArrayGatherV3:0'
                # input_map = {
                #     feed_tensor_name: image_placeholder
                # }
                # data = get_image_content(True)
                feed_tensor_name = 'map/TensorArrayStack/TensorArrayGatherV3:0'
                data = get_image_content(True)
            else:
                # Get input data
                feed_tensor_name = 'ParseExample/ParseExample:0'
                data = get_image_content(False)
            tf.saved_model.loader.load(sess, [tag_set], saved_model_dir, input_map=input_map)
            sess.graph.finalize()
            input_feed_dict = {feed_tensor_name: [data]}
            for i in xrange(num_iterations):
                outputs = sess.run(output_tensor_names, feed_dict=input_feed_dict)
                if print_results:
                    for i, output in enumerate(outputs):
                        print('Result for output key %s:\n%s' % (output_tensor_names[i], output))

    # Gather additional results
    results = dict()
    iter_times = np.array(logger_hook.iter_times[num_warmup_iterations:])
    results['total_time'] = float(np.sum(iter_times))
    results['throughput'] = (batch_size * len(iter_times) * 1.0) / results['total_time']
    results['latency_mean_batch'] = np.mean(iter_times) * 1000
    results['percentile_99th_batch'] = np.percentile(iter_times, q=99, interpolation='lower') * 1000
    for k in results:
        results[k] = float(results[k])
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--saved_model_dir', type=str, required=True,
                        help='Directory containing model checkpoint. If not provided, a ' \
                             'checkpoint may be downloaded automatically and stored in ' \
                             '"{--default_models_dir}/{--model}" for future use.')
    parser.add_argument('--signature_def_key', type=str, required=True, help='')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of images per batch.')
    parser.add_argument('--num_iterations', type=int, default=None,
                        help='How many iterations(batches) to evaluate. If not supplied, the whole set will be evaluated.')
    parser.add_argument('--display_every', type=int, default=100,
                        help='Number of iterations executed between two consecutive display of metrics')
    parser.add_argument('--num_warmup_iterations', type=int, default=50,
                        help='Number of initial iterations skipped from timing')
    parser.add_argument('--profile', action='store_true',
                        help='')
    parser.add_argument('--profile_output_dir', type=str, default="trace_log",
                        help='')
    parser.add_argument('--profile_threshold_ms', type=float, default=30)
    parser.add_argument('--profile_sample_steps', type=int, default=10)
    parser.add_argument('--only_run_backbone', action='store_true', default=False),
    parser.add_argument('--print_results', action='store_true', default=False)
    args = parser.parse_args()

    # Evaluate model
    print('running inference...')
    results = run(
        saved_model_dir=args.saved_model_dir,
        signature_def_key=args.signature_def_key,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        display_every=args.display_every,
        only_run_backbone=args.only_run_backbone,
        print_results=args.print_results,
        enable_profile=args.profile,
        profile_threshold_ms=args.profile_threshold_ms,
        profile_output_dir=args.profile_output_dir,
        profile_sample_steps=args.profile_sample_steps)

    # Display results
    print('results:')
    print('    throughput: %.2f' % results['throughput'])
    print('    latency_mean_batch(ms): %.1f' % results['latency_mean_batch'])
    print('    percentile_99th_batch(ms): %.4f' % results['percentile_99th_batch'])
    print('    total_time(s): %.1f' % results['total_time'])
