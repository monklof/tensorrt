# -*- coding: utf-8 -*-
"""
Created on 1/15/18

@author: zhoulinyuan 
"""


import os.path

# This is a placeholder for a Google-internal import.

import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets import resnet_v2
from preprocessing import  vgg_preprocessing as vgg

tf.app.flags.DEFINE_string('checkpoint_dir', '/opt/zhoulinyuan/inception_v4',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', '/tmp/inception_v4_porn_output',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 4,
                            """Version number of the model.""")
tf.app.flags.DEFINE_integer('image_size', 224,
                            """Needs to provide same value as in training.""")
FLAGS = tf.app.flags.FLAGS

NUM_CLASSES = 3
NUM_TOP_CLASSES = 3

# WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
# SYNSET_FILE = os.path.join(WORKING_DIR, 'imagenet_lsvrc_2015_synsets.txt')
# METADATA_FILE = os.path.join(WORKING_DIR, 'imagenet_metadata.txt')


def export():
  # Create index->synset mapping
  synsets = []
  # with open(SYNSET_FILE) as f:
  #   synsets = f.read().splitlines()
  # # Create synset->metadata mapping
  # texts = {}
  # with open(METADATA_FILE) as f:
  #   for line in f.read().splitlines():
  #     parts = line.split('\t')
  #     assert len(parts) == 2
  #     texts[parts[0]] = parts[1]

  with tf.Graph().as_default():
    # Build inference model.
    # Please refer to Tensorflow inception model for details.

    # Input transformation.
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {
        'image/encoded': tf.FixedLenFeature(
            shape=[], dtype=tf.string),
    }
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    jpegs = tf_example['image/encoded']
    images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

    # Run inference.
    # logits, _ = inception_model.inference(images, NUM_CLASSES + 1)

    # Run inference.
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      logits, _ = resnet_v2.resnet_v2_50(images, NUM_CLASSES, is_training=False)
    logits = tf.nn.softmax(logits)

    # Transform output to topK result.
    values, indices = tf.nn.top_k(logits, NUM_TOP_CLASSES)

    # Create a constant string Tensor where the i'th element is
    # the human readable class description for the i'th index.
    # Note that the 0th index is an unused background class
    # (see inception model definition code).
    # class_descriptions = ['unused background']
    # for s in synsets:
    #   class_descriptions.append(texts[s])
    class_descriptions = ['0_notporn', '1_fang', '2_yuan']
    class_tensor = tf.constant(class_descriptions)

    table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
    classes = table.lookup(tf.to_int64(indices))

    # Restore variables from training checkpoint.
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     inception_model.MOVING_AVERAGE_DECAY)
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     0.9997)
    # variables_to_restore = variable_averages.variables_to_restore()
    # saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()
    with tf.Session() as sess:
      # Restore variables from training checkpoints.
      saver.restore(sess, FLAGS.checkpoint_dir)
      # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      # if ckpt and cdependency_optimizerkpt.model_checkpoint_path:
      #   saver.restore(sess, ckpt.model_checkpoint_path)
      #   # Assuming model_checkpoint_path looks something like:
      #   #   /my-favorite-path/imagenet_train/model.ckpt-0,
      #   # extract global_step from it.
      #   global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      #   print 'Successfully loaded model from %s at step=%s.' % (
      #       ckpt.model_checkpoint_path, global_step)
      # else:
      #   print 'No checkpoint file found at %s' % FLAGS.checkpoint_dir
      #   return

      # keys = sess.graph.get_all_collection_keys()
      sess.graph.clear_collection('resnet_v2_50/_end_points')

      # Export inference model.
      output_path = os.path.join(
          tf.compat.as_bytes(FLAGS.output_dir),
          tf.compat.as_bytes(str(FLAGS.model_version)))
      print 'Exporting trained model to', output_path
      builder = tf.saved_model.builder.SavedModelBuilder(output_path)

      # Build the signature_def_map.
      classify_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
          serialized_tf_example)
      classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(
          classes)
      scores_output_tensor_info = tf.saved_model.utils.build_tensor_info(values)

      classification_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={
                  tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                      classify_inputs_tensor_info
              },
              outputs={
                  tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                      classes_output_tensor_info,
                  tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                      scores_output_tensor_info
              },
              method_name=tf.saved_model.signature_constants.
              CLASSIFY_METHOD_NAME))

      predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(jpegs)
      prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'images': predict_inputs_tensor_info},
              outputs={
                  'classes': classes_output_tensor_info,
                  'scores': scores_output_tensor_info
              },
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          ))

      legacy_init_op = tf.group(
          tf.tables_initializer(), name='legacy_init_op')
      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              'predict_images':
                  prediction_signature,
              tf.saved_model.signature_constants.
              DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  classification_signature,
          },
          legacy_init_op=legacy_init_op)

      builder.save()
      print 'Successfully exported model to %s' % FLAGS.output_dir


def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  image = tf.image.decode_jpeg(image_buffer, channels=3)
  # image = vgg._aspect_preserving_resize(image, vgg._RESIZE_SIDE_MAX)
  image = vgg._aspect_preserving_resize(image, vgg._RESIZE_SIDE_MIN)
  image = vgg._central_crop([image], FLAGS.image_size, FLAGS.image_size)[0]
  image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
  image = tf.to_float(image)
  image = vgg._mean_image_subtraction(image, [vgg._R_MEAN, vgg._G_MEAN, vgg._B_MEAN])
  return image


def main(unused_argv=None):
  export()


if __name__ == '__main__':
  tf.app.run()