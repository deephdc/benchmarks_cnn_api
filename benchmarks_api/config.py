# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))
DATA_DIR = /srv/benchmarks_api/data


# Training and predict(deepaas>=0.5.0) arguments as a dict of dicts 
# with the following structure to feed the deepaas API parser:
# (see also get_train_args() )
# { 'arg1' : {'default': 1,       # default value
#             'help': '',         # can be an empty string
#             'required': False   # bool
#             },
#   'arg2' : {'default': 'value1',
#             'choices': ['value1', 'value2', 'value3'],
#             'help': 'multi-choice argument',
#             'required': False
#             }
# }

train_args = { 'model': {'default': 'resnet50',
                        'choices': ['resnet50', 'alexnet'],
                        'help': 'select CNN model for training',
                        'required': False
                        },
               'n_gpus': {'default': 1,
                          'help': 'Number of GPUs to train (one node only)',
                          'required': False
                          }
}

# !!! deepaas>=0.5.0 calls get_test_args() to get args for 'predict'
predict_args = { 'arg2': {'default': 1,
                          'help': '',
                          'required': False
                         },
}
