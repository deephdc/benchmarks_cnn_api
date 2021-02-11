# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under its License. Please, see the LICENSE file
#
"""
Created on Tue Feb  9 13:50:01 2021

@author: vykozlov
"""

import benchmark_cnn as benchmark
# import project's config.py
import benchmarks_cnn_api.config as cfg
import benchmarks_cnn_api.models.model_utils as mutils
import datetime
import os
import shutil

from collections import OrderedDict
from werkzeug.exceptions import BadRequest


def train(kwargs, run_results):
    """Function to perform training in the case of 
    'synthetic'/'dataset' flavor.
    Updates run_results{}
    """
    
    cnn_score = 0.
    # sort the dictionary alphabetically
    run_results['training'] = OrderedDict(sorted(run_results['training'].items(), 
                                                 key=lambda t: t[0]))

    # calculate "GPU memory scale" for the batch_size
    num_local_gpus, gpu_model, gpu_memory = mutils.get_available_gpus()
    m4gb = 4000000000.
    if kwargs['device'] == 'gpu':
        quotient = gpu_memory // m4gb
        remainder = gpu_memory % m4gb
        rest = remainder/m4gb

        if rest > 0.4 and rest <= 0.75:
            memory_scale = quotient + 0.5
        elif rest > 0.75:
            memory_scale = quotient + 1
        else:
            memory_scale = quotient
    else:
        memory_scale = 1.

    print("[DEBUG] GPU Memory scale = {}".format(memory_scale))
    # Setup and run the benchmark model
    for model, batch_size in cfg.MODELS.items():
        print()
        print("[INFO] Testing {} model ...".format(model))
        # Check if the selected network fits the dataset
        kwargs['model'] = model
        # in the case of CPU, use batch_size = 8
        if kwargs['device'] == 'gpu':
            kwargs['batch_size'] = int(batch_size*memory_scale)
        else:
            kwargs['batch_size'] = cfg.BATCH_SIZE_CPU

        if cfg.DATASET != 'synthetic_data':
            mutils.verify_selected_model(kwargs['model'], kwargs['data_name'])
        else:
            mutils.verify_selected_model(kwargs['model'], 'imagenet')

        # Create Train_Run_Dir to store training data        
        timestamp = int(datetime.datetime.timestamp(datetime.datetime.now()))
        Train_Run_Dir = os.path.join(cfg.MODELS_DIR, str(timestamp))
    
        if not os.path.exists(Train_Run_Dir):
            os.makedirs(Train_Run_Dir)
        else:
            raise BadRequest(
                    "Directory to store training results, {}, already exists!"
                    .format(Train_Run_Dir))
    
        kwargs['train_dir'] = Train_Run_Dir
        kwargs['benchmark_log_dir'] = Train_Run_Dir
    
        print("[DEBUG] benchmark kwargs: %s" % (kwargs)) if cfg.DEBUG_MODEL else ''
        params = benchmark.make_params(**kwargs)
        try:
            params = benchmark.setup(params)
            bench = benchmark.BenchmarkCNN(params)
        except ValueError as param_ex:
            raise BadRequest("ValueError in parameter setup: {}. Params: {}".format(param_ex, params))
 
        # Run benchmark and measure total execution time
        bench.print_info()
    
        try:
            bench.run()
        except ValueError as ve:
            raise BadRequest('ValueError in benchmark execution: {}'.format(ve))
    
        # Read training and metric log files and store training results
        training_file = os.path.join(Train_Run_Dir, 'training.log')
        os.rename(os.path.join(Train_Run_Dir, 'benchmark_run.log'), training_file)
        run_parameters = mutils.parse_logfile_training(training_file)

        metric_file = os.path.join(Train_Run_Dir, 'metric.log')
        # it seems, in the case of synthetic_data we need a delay to close metric.log
        mutils.wait_final_read(metric_file, "average_examples_per_sec")
        run_results['training'][model] = {}
        run_results['training'][model].update(run_parameters) 
        run_results['training'][model]['num_epochs'] = kwargs['num_epochs']
        start, end, avg_examples = mutils.parse_metric_file(metric_file)
        print(start, end, avg_examples)
        cnn_score += avg_examples
        start = mutils.timestr_to_stamp(start, cfg.TIME_FORMAT)
        end = mutils.timestr_to_stamp(end, cfg.TIME_FORMAT)
        run_results["training"][model]["average_examples_per_sec"] = avg_examples
        run_results['training'][model]['execution_time_sec'] = end - start

        # if_cleanup = true: delete training directory
        if cfg.IF_CLEANUP:
            shutil.rmtree(Train_Run_Dir)

    run_results['training']['score'] = cnn_score

