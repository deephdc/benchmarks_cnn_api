# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))
DATA_DIR = '/srv/benchmarks_api/data'
MODEL_DIR = '/srv/benchmarks_api/models'

# Training and predict(deepaas>=0.5.0) arguments as a dict of dicts 
train_args = { 'model': {'default': 'alexnet (ImageNet, Cifar10)',
                         'choices': ['googlenet (ImageNet)', 'inception3 (ImageNet)', 'mobilenet (ImageNet)',
                                     'overfeat (ImageNet)', 'resnet50 (ImageNet)', 'resnet152 (ImageNet)',
                                     'vgg16 (ImageNet)', 'vgg19 (ImageNet)', 'resnet56 (Cifar10)', 'resnet110 (Cifar10)',
                                     'alexnet (ImageNet, Cifar10)'],
                         'help': 'CNN model for training. N.B. Models only support specific data sets, given in \
                                  brackets. Synthetic data can only be processed by ImageNet models.',
                         'required': True
                        },
               'num_gpus': {'default': 1,
                            'help': 'Number of GPUs to train on (one node only). If set to zero, CPU is used.',
                            'required': True,
                           },
               'num_epochs': {'default': 5.0,
                              'help': 'Number of epochs to train on (float value, < 1.0 allowed).',
                              'required': False
                             },
               'optimizer': {'default': 'sgd',
                             'choices': ['sgd','momentum','rmsprop','adam'],
                             'help': 'Optimizer to use.',
                             'required': True
                            },
               'dataset': {'default': 'Synthetic data',
                           'choices': ['Synthetic data', 'imagenet', 'cifar10'],
                           'help': 'Dataset to perform training on. Synthetic \
                            data are ramdomly generated ImageNet-like images.',
                           'required': True
                          },
               'batch_size_per_device': {'default': 64,
                                         'help': 'Batch size for each GPU.',
                                         'required': False
                                        },
               'evaluation': {'default': True,
                              'choices': [False, True],
                              'help': 'Perform evaluation after the \
                              benchmark in order to get accuracy results (only \
                              meaningful on real data sets!).',
                              'required': True
                             }
             }

# !!! deepaas>=0.5.0 calls get_test_args() to get args for 'predict'
predict_args = {}
