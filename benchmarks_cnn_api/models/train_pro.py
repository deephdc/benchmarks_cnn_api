# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under its License. Please, see the LICENSE file
#
"""
Created on Tue Feb  9 13:56:44 2021

@author: vykozlov
"""

import datetime
import json
import os
import benchmark_cnn as benchmark
# import project's config.py
import benchmarks_cnn_api.config as cfg
import benchmarks_cnn_api.models.model_utils as mutils
import shutil

from collections import OrderedDict
from werkzeug.exceptions import BadRequest


def train(train_args, kwargs, run_results):
    """Function for training and evalution used in the "pro" flavor
    Example of run_results:
    
    {
      "machine_config": {
        "cpu_info": {
          "num_cores": 4,
          "cpu_info": "Intel(R) Core(TM) i5-6400 CPU @ 2.70GHz",
          "mhz_per_cpu": 2700
        },
        "gpu_info": {
          "count": 1,
          "model": "GeForce GTX 1070",
          "memory": 7811910861
        },
        "memory_total": 16765304832,
        "memory_available": 14814158848
      },
      "benchmark": {
        "version": "0.1.0.dev48",
        "flavor": "pro",
        "docker_base_image": "",
        "dataset": "synthetic_data",
        "tf_version": "1.14.0"
      },
      "training": {
        "allow_growth": true,
        "batch_size": 64,
        "batch_size_per_device": 64,
        "data_format": "NCHW",
        "device": "gpu",
        "local_parameter_device": "cpu",
        "model": "resnet50",
        "num_batches": 100,
        "num_epochs": 0,
        "num_gpus": 1,
        "optimizer": "sgd",
        "use_fp16": false,
        "variable_update": "parameter_server",
        "weight_decay": 0.00004,
        "result": {
          "average_examples_per_sec": 124.41983172966508,
          "execution_start_time": "2021-02-10T22:59:17.434987Z",
          "execution_end_time": "2021-02-10T23:00:08.358017Z",
          "execution_time_sec": 50.92302989959717
        }
      },
      "evaluation": {
        "batch_size": 64,
        "batch_size_per_device": 64,
        "data_format": "NCHW",
        "device": "gpu",
        "model": "resnet50",
        "num_batches": 100,
        "num_gpus": 1,
        "result": {
          "average_examples_per_sec": 401.17907755615994,
          "top_1_accuracy": 0.0015625,
          "top_5_accuracy": 0.00609375
        }
      },
      "global_start_time": "2021-02-10T22:59:01.664787Z",
      "global_end_time": "2021-02-10T23:00:30.489734Z",
      "global_execution_time_sec": 88.82494688034058
    }        
    """

    # Add more training arguments
    kwargs['batch_size'] = train_args['batch_size_per_device']
    kwargs['model'] = train_args['model'].split(' ')[0]
    kwargs['weight_decay'] = train_args['weight_decay']
    
    # Log additional arguments in run_results[]
    run_results['training']['model'] = kwargs['model']
    run_results["training"]['num_epochs'] = kwargs['num_epochs']
    run_results['training']['weight_decay'] = kwargs['weight_decay']

    # Check if the selected network fits the dataset
    dataset_name = ( kwargs['data_name'] if 'data_name' in kwargs.keys() 
                                         else 'synthetic_data' )
    if dataset_name != 'synthetic_data':
        mutils.verify_selected_model(kwargs['model'], kwargs['data_name'])
    else:
        mutils.verify_selected_model(kwargs['model'], 'imagenet')

    # Create Train_Run_Dir to store training data
    timestamp = int(datetime.datetime.timestamp(datetime.datetime.now()))
    Train_Run_Dir = os.path.join(cfg.MODELS_DIR, str(timestamp))
    Eval_Dir = os.path.join(Train_Run_Dir, "eval_dir")

    if not os.path.exists(Train_Run_Dir):
        os.makedirs(Train_Run_Dir)
    else:
        raise BadRequest(
                "Directory to store training results, {}, already exists!"
                .format(Train_Run_Dir))

    kwargs['train_dir'] = Train_Run_Dir
    kwargs['benchmark_log_dir'] = Train_Run_Dir

    # Log training directories, if they are not deleted later
    if not train_args['if_cleanup']:
        run_results['training']['train_dir'] = kwargs['train_dir']
        run_results['training']['benchmark_log_dir'] = kwargs['benchmark_log_dir']

    # In kwargs num_gpus=1 also for CPU, update num_gpus in run_results to 0
    if run_results["training"]["device"] == "cpu":
        run_results["training"]["num_gpus"] = 0 # avoid misleading info

    # Setup and run the benchmark model
    print("[DEBUG] benchmark kwargs: %s" % (kwargs)) if cfg.DEBUG_MODEL else ''
    params = benchmark.make_params(**kwargs)
    try:
        params = benchmark.setup(params)
        bench = benchmark.BenchmarkCNN(params)
    except ValueError as param_ex:
        raise BadRequest("ValueError in parameter setup: {}. Params: {}".format(param_ex, params))

    # Run benchmark
    bench.print_info()
    try:
        bench.run()
    except ValueError as ve:
        raise BadRequest('ValueError in benchmark execution: {}'.format(ve))

    # Read training and metric log files and store training results
    training_file = os.path.join(Train_Run_Dir, 'training.log')
    os.rename(os.path.join(Train_Run_Dir, 'benchmark_run.log'), training_file)
    run_parameters = mutils.parse_logfile_training(training_file)
    run_results['training'].update(run_parameters)

    # sort the dictionary alphabetically
    run_results['training'] = OrderedDict(sorted(run_results['training'].items(), 
                                          key=lambda t: t[0]))

    metric_file = os.path.join(Train_Run_Dir, 'metric.log')
    # it seems, in the case of synthetic_data we need a delay to close metric.log
    mutils.wait_final_read(metric_file, "average_examples_per_sec")
    start, end, avg_examples = mutils.parse_metric_file(metric_file)
    run_results['training']['result'] = {}
    run_results["training"]["result"]["average_examples_per_sec"] = avg_examples
    run_results['training']['result']['execution_start_time'] = start
    run_results['training']['result']['execution_end_time'] = end
    start_sec = mutils.timestr_to_stamp(start, cfg.TIME_FORMAT)
    end_sec = mutils.timestr_to_stamp(end, cfg.TIME_FORMAT)
    run_results['training']['result']['execution_time_sec'] = end_sec - start_sec

    ## Evaluation ##
    if train_args['evaluation']:
        run_results["evaluation"] = {}

        kwargs_eval = {'model': kwargs['model'],
                       'num_gpus': kwargs['num_gpus'],
                       'device': kwargs['device'],
                       'data_format': kwargs['data_format'],
                       'benchmark_log_dir': kwargs['benchmark_log_dir'],
                       'train_dir': kwargs['train_dir'],
                       'eval': True
                       # 'eval_dir': Eval_Dir,
                       }
        run_results['evaluation']['device'] = kwargs_eval['device']
        if run_results['evaluation']['device'] == 'gpu':
            run_results['evaluation']['num_gpus'] = kwargs_eval['num_gpus']  # only for GPU to avoid confusion

        # Locate data
        if dataset_name != 'synthetic_data':
            kwargs_eval['data_name'] = kwargs['data_name']
            kwargs_eval['data_dir'] = kwargs['data_dir']

        # Setup and run the evaluation
        params_eval = benchmark.make_params(**kwargs_eval)
        try:
            params_eval = benchmark.setup(params_eval)
            evaluation = benchmark.BenchmarkCNN(params_eval)
        except ValueError as param_ex:
            raise BadRequest("ValueError: {}".format(param_ex))

        evaluation.print_info()
        evaluation.run()

        # Read log files and get evaluation results
        evaluation_file = os.path.join(Train_Run_Dir, 'evaluation.log')
        os.rename(os.path.join(Train_Run_Dir, 'benchmark_run.log'), evaluation_file)
        run_parameters = mutils.parse_logfile_evaluation(evaluation_file)
        run_results['evaluation'].update(run_parameters)

        # sort the dictionary alphabetically
        run_results['evaluation'] = OrderedDict(sorted(run_results['evaluation'].items(), 
                                                       key=lambda t: t[0]))

        logfile = os.path.join(Train_Run_Dir, 'metric.log')
        run_results['evaluation']['result'] = {}

        # it seems, in the case of synthetic_data we need a delay to close evaluation.log
        mutils.wait_final_read(logfile, "eval_average_examples_per_sec")
        
        with open(logfile, "r") as f:
            for line in f:
                l = json.loads(line)
                if l["name"] == "eval_average_examples_per_sec":
                    run_results["evaluation"]['result']["average_examples_per_sec"] = l["value"]
                if l["name"] == "eval_top_1_accuracy":
                    run_results["evaluation"]['result']["top_1_accuracy"] = l["value"]
                if l["name"] == "eval_top_5_accuracy":
                    run_results["evaluation"]['result']["top_5_accuracy"] = l["value"]


    if train_args['if_cleanup']:
        shutil.rmtree(Train_Run_Dir)