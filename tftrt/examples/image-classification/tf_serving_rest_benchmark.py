# Copyright 2018 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model

The SavedModel must be one that can take JPEG images as inputs.

Typical usage example:

    tf_serving_rest_benchmark.py
"""

from __future__ import print_function

import base64
import requests
import sys
import os
import os.path


# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = '%s/v1/models/resnet:predict' % sys.argv[1]

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
  content = get_image_content()

  # Compose a JSON Predict request (send JPEG image in base64).
  jpeg_bytes = base64.b64encode(content).decode('utf-8')
  predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes

  # Send few requests to warm-up the model.
  for _ in range(3):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()

  # Send few actual requests and report average latency.

  interval = int(sys.argv[3])
  num_requests = int(sys.argv[2])
  loop = num_requests/interval

  for __ in range(loop):
    total_time = 0
    for _ in range(interval):
      response = requests.post(SERVER_URL, data=predict_request)
      response.raise_for_status()
      total_time += response.elapsed.total_seconds()
      prediction = response.json()['predictions'][0]
    print('Prediction class: %s, avg latency: %.2f ms, avg request/second: %.2f' % (
      prediction['classes'], (total_time*1000)/interval, (interval/total_time)))


if __name__ == '__main__':
  main()
