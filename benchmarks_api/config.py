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
train_args = { 'model': {'default': 'alexnet',
                        'choices': ['alexnet', 'resnet'],
                        'help': 'CNN model for training. ResNet model is chosen \
                        according to dataset (ImageNet - ResNet50, Cifar10 - \
                        ResNet56).',
                        'required': False
                        },
               'num_gpus': {'default': 1,
                          'help': 'Number of GPUs to train (one node only).',
                          'required': False
                         },
               'num_epochs': {'default': 5.0,
                              'help': 'Number of epochs to train on.',
                              'required': False
                             },
               'optimizer': {'default': 'sgd',
                             'choices': ['sgd','momentum','rmsprop','adam'],
                             'help': 'Optimizer to use.',
                             'required': False
                            },
               'dataset': {'default': 'Synthetic data',
                           'choices': ['Synthetic data', 'ImageNet', 'Cifar10'],
                           'help': 'Dataset to perform training on. Synthetic \
                            data are ramdomly generated ImageNet-like images.',
                           'required': False
                          },
               'batch_size': {'default': 64,
                              'help':'Batch size for each GPU.',
                              'required': False
                             }
             }

# !!! deepaas>=0.5.0 calls get_test_args() to get args for 'predict'
predict_args = {'model': {'default': 'alexnet',
                          'choices': ['alexnet', 'resnet50'],
                          'help': 'CNN model for training.',
                          'required': False
                         },
               'num_gpus': {'default': 1,
                            'help': 'Number of GPUs to train (one node only).',
                            'required': False
                           },
               'dataset': {'default': 'Synthetic data',
                           'choices': ['Synthetic data', 'ImageNet', 'Cifar10'],
                           'help': 'Dataset to perform training on. Synthetic \
                            data are ramdomly generated ImageNet-like images.',
                           'required': False
                          }
               }
