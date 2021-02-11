# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

import os
from webargs import fields
from marshmallow import Schema, INCLUDE
from official.utils.logs import logger as official_logger

# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_in_out_base_dir) + \
        "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        print(msg)

DATA_DIR = os.path.join(IN_OUT_BASE_DIR, 'data')
MODELS_DIR = os.path.join(IN_OUT_BASE_DIR, 'models')

CIFAR10_REMOTE_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
IMAGENET_MINI_REMOTE_URL="https://nc.deep-hybrid-datacloud.eu/s/aZr8Hi5Jk7GMSe4/download?path=%2F&files=imagenet_mini.tar"

# FLAAT needs a list of trusted OIDC Providers.
# we select following three providers:
Flaat_trusted_OP_list = [
'https://aai.egi.eu/oidc/',
'https://iam.deep-hybrid-datacloud.eu/',
'https://iam.extreme-datacloud.eu/',
]

# Define CONSTANTS for this benchmark flavors:
# BENCHMARK_FLAVOR: ['synthetic', 'dataset', 'accuracy', 'pro']
BENCHMARK_FLAVOR = os.getenv('BENCHMARK_FLAVOR', 'pro')
# DATASET options: ['synthetic_data', 'imagnet_mini', 'imagenet']
if BENCHMARK_FLAVOR == 'synthetic':
    DATASET = 'synthetic_data'
elif BENCHMARK_FLAVOR == 'dataset':
    DATASET = 'imagenet_mini'
elif BENCHMARK_FLAVOR == 'accuracy':
    DATASET = 'imagenet'
else:
    DATASET = 'synthetic_data'
    BENCHMARK_FLAVOR = 'pro'
    
DOCKER_BASE_IMAGE = os.getenv('DOCKER_BASE_IMAGE', '')

# Use the timeformat of tf-benchmark, smth. '%Y-%m-%dT%H:%M:%S.%fZ'
TIME_FORMAT = official_logger._DATE_TIME_FORMAT_PATTERN
## BENCHMARK version ['synthetic', 'dataset', 'accuracy']
## needs following to be defined.    
# experimentally found:
# GTX1070, 8GB:
# ============
# googlenet: 192 
# imagenet : 48
# resnet50 : 48
# vgg16    : 32 (48 - reboot)
# => score = ca. 600
#
# also:
# mobilenet: 1024
# resnet152: 24
# vgg19    : 32 (48 - reboot)
##
# batch_size(s) are for 4GB memory:
MODELS = {'googlenet' : 96,
          'inception3' : 24,
          'resnet50' : 24,
          'vgg16': 16
          }
BATCH_SIZE_CPU = 16
NUM_EPOCHS = 0
OPTIMIZER = 'sgd'  # to consider: [sgd','momentum','rmsprop','adam']
USE_FP16 = False
EVALUATION = False
IF_CLEANUP = True
##

## DEBUG Flags
DEBUG_MODEL = True

# Training and predict(deepaas>=0.5.0) arguments as a dict of dicts 

def get_train_args_schema():
    """Function to return the reference to proper TrainArgsSchema
    """
    
    if ( BENCHMARK_FLAVOR == 'synthetic' or 
         BENCHMARK_FLAVOR == 'dataset' or
         BENCHMARK_FLAVOR == 'accuracy'
        ):
        train_args_schema = TrainArgsSchemaBench()
    else:
        train_args_schema = TrainArgsSchemaPro()
    
    return train_args_schema
    

# class / place to describe arguments for train()
class TrainArgsSchemaBench(Schema):
    class Meta:
        unknown = INCLUDE  # supports extra parameters

    num_gpus = fields.Integer(missing=1,
                              description='Number of GPUs to train on \
                              (one node only). If set to zero, CPU is used.',
                              required= False
                              )

# 'pro' version of class TranArgsSchema
class TrainArgsSchemaPro(Schema):
    class Meta:
        unknown = INCLUDE  # supports extra parameters

    batch_size_per_device = fields.Integer(missing=64,
                                           description='Batch size for each GPU.',
                                           required= False
                                           )
    dataset = fields.Str(missing='synthetic_data',
                         enum=['synthetic_data', 
                               'imagenet',
                               'imagenet_mini',
                               'cifar10'],
                         description='Dataset to perform training on. \
                         synthetic_data: randomly generated ImageNet-like \
                         images; imagenet_mini: 3% of the real ImageNet \
                         dataset',
                         required=False
                         )
    model = fields.Str(missing='resnet50 (ImageNet)',
                       enum = ['googlenet (ImageNet)',
                               'inception3 (ImageNet)',
                               'mobilenet (ImageNet)',
                               'overfeat (ImageNet)',
                               'resnet50 (ImageNet)',
                               'resnet152 (ImageNet)',
                               'vgg16 (ImageNet)',
                               'vgg19 (ImageNet)',
                               'resnet56 (Cifar10)',
                               'resnet110 (Cifar10)',
                               'alexnet (ImageNet, Cifar10)'],
                       description='CNN model for training. N.B. Models only \
                       support specific data sets, given in brackets. \
                       synthetic_data can only be processed by ImageNet models.',
                       required=False
                       )
    num_gpus = fields.Integer(missing=1,
                              description='Number of GPUs to train on \
                              (one node only). If set to zero, CPU is used.',
                              required= False
                              )
    num_epochs = fields.Float(missing=1.0,
                              description='Number of epochs to \
                              train on (float value, < 1.0 allowed).',
                              required= False
                              )
    optimizer = fields.Str(missing='sgd',
                           enum=['sgd','momentum','rmsprop','adam'],
                           description='Optimizer to use.',
                           required= False 
                           )
    use_fp16 = fields.Boolean(missing=False,
                              enum = [False, True],
                              description='Use 16-bit floats for certain \
                              tensors instead of 32-bit floats. ',
                              required=False
                              )
    weight_decay = fields.Float(missing=4.0e-5,
                              description='Weight decay factor for training',
                              required=False
                              )    
    evaluation = fields.Boolean(missing=True,
                                enum = [False, True],
                                description='Perform evaluation after the \
                                benchmark in order to get accuracy results \
                                (only meaningful on real data sets!).',
                                required=False
                                )
    if_cleanup = fields.Boolean(missing=False,
                              enum = [False, True],
                              description='If to delete training and \
                              evaluation directories.',
                              required=False
                              )


# class / place to describe arguments for predict()
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # supports extra parameters

    files = fields.Field(required=False,
                         missing=None,
                         type="file",
                         data_key="data",
                         location="form",
                         description="NOT implemented!."
                        )
    
