# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under its License. Please, see the LICENSE file
#
"""
Created on Sat Feb 13 22:39:18 2021

@author: vykozlov
"""

import benchmarks_cnn_api.config as cfg
import benchmarks_cnn_api.models.deep_api as deep_api
import os
import unittest

class TestModelFunc(unittest.TestCase):
    def setUp(self):
        self.MODELS = { 'mobilenet' : 2 }
        self.BATCH_SIZE_CPU = 2
        self.NUM_EPOCHS = 0.
        self.USE_FP16 = False
        self.train_args = {
            'dataset': 'synthetic_data',
            'num_gpus': 0,
            'num_epochs': self.NUM_EPOCHS,
            'batch_size_per_device': self.BATCH_SIZE_CPU,
            'model': 'googlenet (ImageNet)', # mobilenet, googlenet
            'optimizer': 'sgd',
            'use_fp16': self.USE_FP16,
            'weight_decay': 1e-4,
            'evaluation': True,
            'if_cleanup': True
        }
        ### might be useful in future...:
        #self.kwargs = {
        #    'batch_size': self.train_args['batch_size_per_device'],
        #    'model': self.train_args['model'].split(' ')[0],
        #    'num_gpus': self.train_args['num_gpus'],
        #    'num_epochs': self.train_args['num_epochs'],
        #    'optimizer': self.train_args['optimizer'],
        #    'use_fp16': self.train_args['use_fp16'],
        #    'weight_decay': self.train_args['weight_decay'],
        #    'local_parameter_device': 'cpu',
        #    'variable_update': 'parameter_server',
        #    'allow_growth': True,
        #    'device': 'cpu',
        #    'data_format': 'NHWC' ,
        #    'print_training_accuracy': True,          
        #}
        ###
        #self.run_results = {
        #    'machine_config': {},
        #    'benchmark': {
        #        'version': deep_api.get_metadata()['Version'],
        #        'flavor': 'synthetic',
        #        'docker_base_image': cfg.DOCKER_BASE_IMAGE,
        #        'dataset' : self.train_args['dataset'],
        #        'tf_version': ''
        #    },
        #   'training': {
        #       'num_gpus': self.kwargs['num_gpus'],
        #       'optimizer': self.kwargs['optimizer'],
        #       'use_fp16': self.kwargs['use_fp16'],
        #       'local_parameter_device': self.kwargs['local_parameter_device'],
        #       'variable_update': self.kwargs['variable_update'],
        #       'allow_growth': self.kwargs['allow_growth'],
        #       'device': self.kwargs['device'],
        #       'data_format': self.kwargs['data_format']
        #   },
        #}
        ###

    def test_train_pro(self):
        """Function to test the 'pro' flavor
        """
        #self.kwargs['num_gpus'] = 1  # Important: tensorflow uses this also to specify the number of CPUs
        cfg.BENCHMARK_FLAVOR = 'pro'
        cfg.BATCH_SIZE_CPU = self.BATCH_SIZE_CPU
        run_results = deep_api.train(**self.train_args)
        train_avg_examples = run_results["training"]["result"]["average_examples_per_sec"]
        model = run_results['training']['model']
        eval_avg_examples = run_results["evaluation"]["result"]["average_examples_per_sec"]
        self.assertTrue(train_avg_examples > 0.)
        self.assertEqual(model, 'googlenet') # mobilenet, googlenet
        self.assertTrue(eval_avg_examples > 0.)

    def test_train_synth(self):
        """Function to test synthetic/dataset flavor
        """
        #self.kwargs['num_gpus'] = 1  # Important: tensorflow uses this also to specify the number of CPUs
        cfg.BENCHMARK_FLAVOR = 'synthetic'
        cfg.MODELS = self.MODELS
        cfg.BATCH_SIZE_CPU = self.BATCH_SIZE_CPU
        cfg.NUM_EPOCHS = self.NUM_EPOCHS
        cfg.USE_FP16 = self.USE_FP16
        self.train_args['evaluation'] = False
        model = list(self.MODELS.keys())[0]
        run_results = deep_api.train(**self.train_args)
        train_model_avg_examples = run_results["training"][model]["average_examples_per_sec"]
        score = run_results['training']['score']
        self.assertTrue(train_model_avg_examples > 0.)
        self.assertTrue(score > 0.)


if __name__ == '__main__':
    unittest.main()

