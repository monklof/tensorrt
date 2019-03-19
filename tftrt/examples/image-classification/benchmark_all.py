#! /usr/bin/env python
# coding: utf-8

from __future__ import print_function

import os
import sys
import subprocess
import json

def benchmark(model="resnet_v1_50", **kwargs):
    cmd = "python image_classification.py --model %(model)s " \
          "--data_dir /workdir/workspace/models/research/slim/imagenet-data/validation/  " \
          "--calib_data_dir /workdir/workspace/models/research/slim/imagenet-data/validation-as-train/ " \
          "--batch_size %(batch_size)s --num_iterations %(num_iterations)s " \
          "--display_every %(display_every)s --precision %(precision)s " \
          "--minimum_segment_size 5 %(flags)s --remove_accuracy"

    params = kwargs.copy()
    params['model'] = model
    params['flags'] = ""
    if kwargs['use_synthetic']:
        params['flags'] += ' --use_synthetic'
    if kwargs['use_trt']:
        params['flags'] += ' --use_trt'

    cmd = cmd % params
    cmd = "echo `date` >> ./benchmark.log; echo '%s' >> ./benchmark.log; %s >> ./benchmark.log 2>&1 " % (cmd, cmd)
    ret_code = os.system(cmd)
    default_result = dict(
            platform="TF-TRT" if kwargs['use_trt'] else "TensorFlow",
            use_synthetic=kwargs['use_synthetic'],
            precision=kwargs['precision'],
            batch_size=kwargs['batch_size'],
            throughput=-1,
            accuracy=-1,
            latency_mean_batch=-1,
            percentile_99th_batch=-1)
    if ret_code == 0:
        results = json.loads(subprocess.check_output("tail -1 ./benchmark.log", shell=True))
        default_result.update({k: results[k] for k in (
            'throughput', 'accuracy',
            'latency_mean_batch', 'percentile_99th_batch')})
    return default_result


def gen_benchmark_cases():
    batch_size_choices = (1, 2, 4, 8, 16, 32, 64, 128)
    num_iterations_choices = (1000, 1000, 1000, 1000, 500, 500, 500, 500)
    display_every_choices = (100, 100, 100, 100, 50, 50, 50, 50)

    def gen_case(use_synthetic, use_trt, precision):
        cases = []
        for i in xrange(len(batch_size_choices)):
            cases.append(dict(use_synthetic=use_synthetic,
                              use_trt=use_trt, precision=precision,
                              batch_size=batch_size_choices[i],
                              num_iterations=num_iterations_choices[i],
                              display_every=display_every_choices[i]))
        return cases

    cases = []

    # run in native mode using image-net validation data
    cases += gen_case(use_synthetic=False, use_trt=False, precision='fp32')
    # run in tf-trt mode (fp32) using image-net validation data
    cases += gen_case(use_synthetic=False, use_trt=True, precision='fp32')
    # run in tf-trt mode (fp16) using image-net validation data
    cases += gen_case(use_synthetic=False, use_trt=True, precision='fp16')
    # # run in tf-trt mode (int8) using image-net validation data
    # cases += gen_case(use_synthetic=False, use_trt=True, precision='int8')

    # run in native mode in synthetic mode
    cases += gen_case(use_synthetic=True, use_trt=False, precision='fp32')
    # run in tf-trt mode (fp32) in synthetic mode
    cases += gen_case(use_synthetic=True, use_trt=True, precision='fp32')
    # run in tf-trt mode (fp16) in synthetic mode
    cases += gen_case(use_synthetic=True, use_trt=True, precision='fp16')
    # # run in tf-trt mode (int8) in synthetic mode
    # cases += gen_case(use_synthetic=True, use_trt=True, precision='int8')

    return cases


def benchmark_main():
    cases = gen_benchmark_cases()
    results = []
    i = 0
    for case in cases:
        i += 1
        print('[%d/%d] running benchmark: %s' % (i, len(cases), case))
        result = benchmark(model=sys.argv[1], **case)
        print('----> %s' % result)
        results.append(result)

    keys = ['platform',
            'use_synthetic',
            'precision',
            'batch_size',
            'throughput',
            'latency_mean_batch',
            'percentile_99th_batch',
            'accuracy']
    print('\t'.join(keys))
    for result in results:
        print('\t'.join([str(result[k]) for k in keys]))


if __name__ == "__main__":
    benchmark_main()

