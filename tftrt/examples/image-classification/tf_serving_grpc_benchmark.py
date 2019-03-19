"""
based on: https://github.com/dwyatte/tensorflow-serving-benchmark/blob/master/client/tf_serving_grpc_benchmark.py
"""

import os
import time
import argparse
import threading
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

class Benchmark(object):
    """
    num_requests: Number of requests.
    max_concurrent: Maximum number of concurrent requests.
    """

    def __init__(self, num_requests, max_concurrent, log_interval=10):
        self._num_requests = num_requests
        self._max_concurrent = max_concurrent
        self.done = 0
        self._active = 0
        self.error = 0
        self._condition = threading.Condition()
        self._request_times = [0] * num_requests
        self.log_interval = log_interval

    def inc_done(self, elapsed_time, error=False):
        with self._condition:
            if error:
                self.error += 1
            self._request_times[self.done] = elapsed_time
            self.done += 1
            if not self.done % self.log_interval:
                print('last %s requests avg latency: %.2f ms' % (
                    self.log_interval,
                    (1000.0 * sum(self._request_times[self.done-self.log_interval:self.done]))\
                    / self.log_interval))
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def throttle(self):
        with self._condition:
            while self._active == self._max_concurrent:
                self._condition.wait()
            self._active += 1

    def wait(self):
        with self._condition:
            while self.done < self._num_requests:
                self._condition.wait()

    @property
    def total_elapsed_time(self):
        return sum(self._request_times)


def _create_rpc_callback(benchmark, start_time):
    def _callback(result_future):
        elapsed_time = time.time() - start_time
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            # result = result_future.result().outputs['classes'].int64_val[0]
            result = result_future.result().outputs['classes']
            # print result
        benchmark.inc_done(elapsed_time, error=exception)
        benchmark.dec_active()
    return _callback



def get_image_content():
  # The image URL is the location of the image we should send to the server
  IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
  # Download the image
  if not os.path.exists('cat.jpg'):
    os.system('wget %s' % IMAGE_URL)
  # dl_request = requests.get(IMAGE_URL, stream=True)
  # dl_request.raise_for_status()
  with open('cat.jpg', 'rb') as f:
    content = f.read()

  return content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=os.getenv('MODEL_NAME', None))
    parser.add_argument('--signature_name', default='predict')
    parser.add_argument('--signature_input_key', default='image_bytes')
    parser.add_argument('--serving_host', default=os.getenv('SERVING_HOST', None))
    parser.add_argument('--serving_port', default=os.getenv('SERVING_PORT', '8500'))
    parser.add_argument('--num_requests', default=1000, type=int)
    parser.add_argument('--max_concurrent', default=1, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    args = parser.parse_args()

    channel = grpc.insecure_channel('{}:{}'.format(args.serving_host, args.serving_port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    benchmark = Benchmark(int(args.num_requests), int(args.max_concurrent),
                          log_interval=args.log_interval)

    data = get_image_content()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = args.model_name
    request.model_spec.signature_name = args.signature_name
    # request.inputs[tf.saved_model.signature_constants.PREDICT_INPUTS].CopyFrom(
    #     tf.contrib.util.make_tensor_proto([[i % 2**32]], shape=[1, 1]))
    request.inputs[args.signature_input_key].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=[1]))

    # warm up
    for i in range(args.log_interval):
        result = stub.Predict(request, 120)

    start_time = time.time()
    for i in range(args.num_requests):
        benchmark.throttle()
        result = stub.Predict.future(request, 10)
        result.add_done_callback(_create_rpc_callback(benchmark, time.time()))
    benchmark.wait()
    end_time = time.time()

    print()
    print('{} requests, {} failed, {} success({} max concurrent)'.format(
        args.num_requests, benchmark.error, benchmark.done-benchmark.error,
        args.max_concurrent))
    print('requests/second: %.2f' % (args.num_requests/(end_time-start_time)))
    print('overall avg latency: %.2f ms' % ((end_time-start_time)*1000.0/args.num_requests))
    print('avg latency(plus pending): %.2f ms' % (1000.0 * benchmark.total_elapsed_time/args.num_requests))

if __name__ == "__main__":
    main()