# -*- coding: utf-8 -*-
"""
Model description
"""

import datetime
import os
import pkg_resources
import re
import tempfile

from werkzeug.exceptions import BadRequest
from webargs import fields
# import project's config.py
import benchmarks_cnn_api.config as cfg
import benchmarks_cnn_api.models.model_utils as mutils
import cnn_util
import benchmarks_cnn_api.models.train_synth_or_data as train_sd
import benchmarks_cnn_api.models.train_pro as train_pro

from aiohttp.web import HTTPBadRequest
from collections import OrderedDict

from functools import wraps


## Authorization
from flaat import Flaat
flaat = Flaat()
# DEEP API V2 uses aiohttp, thus hard code
# NB: currenlty aiohttp is not fully supported by flaat!
flaat.set_web_framework("aiohttp")
flaat.set_trusted_OP_list(cfg.Flaat_trusted_OP_list)

TMP_DIR = tempfile.gettempdir() # set the temporary directory

# assing it globally
num_local_gpus, gpu_model, gpu_memory = mutils.get_available_gpus()

def _catch_error(f):
    """Decorate function to return an error as HTTPBadRequest, in case
    """
    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap


def _fields_to_dict(fields_in):
    """Function to convert mashmallow fields to dict()"""

    dict_out = {}

    for key, val in fields_in.items():
        param = {}
        param['default'] = val.missing
        param['type'] = type(val.missing)
        if key == 'files' or key == 'urls':
            param['type'] = str

        val_help = val.metadata['description']
        # argparse hates % sign:
        if '%' in val_help:
            # replace single occurancies of '%' with '%%'
            # since '%%' is accepted by argparse
            val_help = re.sub(r'(?<!%)%(?!%)', r'%%', val_help)

        if 'enum' in val.metadata.keys():
            val_help = "{}. Choices: {}".format(val_help,
                                                val.metadata['enum'])
        param['help'] = val_help

        try:
            val_req = val.required
        except Exception:
            val_req = False
        param['required'] = val_req

        dict_out[key] = param
    return dict_out


def get_metadata():
    """
    Function to read metadata
    """

    module = __name__.split('.', 1)

    pkg = pkg_resources.get_distribution(module[0])
    meta = {
        'Name': None,
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': None,
        'Author-email': None,
        'License': None,
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if line.startswith(par + ":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta


def predict_data(*args):
    """
    Function to make prediction on a local file
    """
    message = 'Not implemented (predict_file())'
    message = {"Error": message}
    return message


def predict_url(*args):
    """
    Function to make prediction on a URL
    """
    message = 'Not implemented (predict_url())'
    message = {"Error": message}
    return message


@_catch_error
def predict(**kwargs):
    
    print("predict(**kwargs) - kwargs: %s" % (kwargs)) if cfg.DEBUG_MODEL else ''

    if (not any([kwargs['urls'], kwargs['files']]) or
            all([kwargs['urls'], kwargs['files']])):
        raise Exception("You must provide either 'url' or 'data' in the payload")

    if kwargs['files']:
        kwargs['files'] = [kwargs['files']]  # patch until list is available
        return predict_data(kwargs)
    elif kwargs['urls']:
        kwargs['urls'] = [kwargs['urls']]  # patch until list is available
        return predict_url(kwargs)


###
# To allow only authorized people to do training,
# uncomment the following line (NB: currenlty aiohttp is not supported!!)
#@flaat.login_required()
def train(**train_kwargs):
    """
    Train network
    train_args : dict
    """

    print("[DEBUG] train(**train_kwargs) - train_kwargs: %s" % (train_kwargs)) if cfg.DEBUG_MODEL else ''

    # use the schema
    schema = cfg.get_train_args_schema()
    # deserialize key-word arguments
    train_args = schema.load(train_kwargs)
    train_keys = train_args.keys()
    # log the dataset name
    # dataset options: ['synthetic_data', 'imagnet_mini', 'imagenet']
    dataset_name = 'synthetic_data'
    if cfg.BENCHMARK_TYPE == 'benchmark':
        if train_args['flavor'] == 'synthetic':
            benchmark_flavor = 'synthetic'
            dataset_name = 'synthetic_data'
        if train_args['flavor'] == 'dataset':
            benchmark_flavor = 'dataset'
            dataset_name = 'imagenet_mini'

    if cfg.BENCHMARK_TYPE == 'pro':
        benchmark_flavor = 'pro'
        dataset_name = train_args['dataset']  

    # log the Tensorflow version
    tf_version = '.'.join([str(x) for x in cnn_util.tensorflow_version_tuple()])

    # Declare training arguments for tf_cnn_benchmarks.
    # Defaults are from config.py
    kwargs = {}
    kwargs['num_gpus'] = train_args['num_gpus'] if 'num_gpus' in train_keys else 0
    kwargs['num_epochs'] = ( train_args['num_epochs'] if 'num_epochs' in train_keys 
                                                      else cfg.NUM_EPOCHS )
    kwargs['optimizer'] = ( train_args['optimizer'] if 'optimizer' in train_keys 
                                                    else cfg.OPTIMIZER )
    kwargs['use_fp16'] = ( train_args['use_fp16'] if 'use_fp16' in train_keys 
                                                      else cfg.USE_FP16 )
    kwargs['local_parameter_device'] = 'cpu'
    kwargs['variable_update'] = cfg.VARIABLE_UPDATE
    kwargs['allow_growth'] = True
    kwargs['print_training_accuracy'] = True
    # how often print training info
    kwargs['display_every'] = 10 if kwargs['num_epochs'] < 1.0 else 100

    # If no GPU is available or the gpu option is set to 0, run CPU mode
    if num_local_gpus == 0 or kwargs['num_gpus'] == 0:
        kwargs['device'] = 'cpu'
        kwargs['data_format'] = 'NHWC'  # cpu data format
        kwargs['num_gpus'] = 1  # Important: tensorflow uses this also to specify the number of CPUs
    else:
        kwargs['device'] = 'gpu'
        kwargs['data_format'] = 'NCHW'

    if dataset_name != 'synthetic_data':
        kwargs['data_name'] = (dataset_name if dataset_name != 'imagenet_mini' 
                                            else 'imagenet')
        kwargs['data_dir'] = os.path.join(cfg.DATA_DIR, dataset_name)

    # Log training info configured for benchmark_cnn in the run_results
    run_results = {'machine_config': {},
                   'benchmark': {
                       'version': get_metadata()['Version'],
                       'flavor': benchmark_flavor,
                       'docker_base_image': cfg.DOCKER_BASE_IMAGE,
                       'dataset' : dataset_name,
                       'tf_version': tf_version
                       },
                   'training': {
                       'num_gpus': 0,
                       'optimizer': '',
                       'use_fp16': '',
                       'local_parameter_device': '',
                       'variable_update': '',
                       'allow_growth': '',
                       'device': '',
                       'data_format': '',
                       'models': []
                       },
                  }

    # Update run_results with values configured for tf_cnn_benchmarks (kwargs)
    results_train_keys = run_results["training"].keys()
    kwargs_keys = kwargs.keys()
    for key in results_train_keys:
        if key in kwargs_keys:
            run_results['training'][key] = kwargs[key] 
    # In kwargs num_gpus=1 also for CPU, update num_gpus in run_results to 0
    if run_results["training"]["device"] == "cpu":
        run_results["training"]["num_gpus"] = 0 # avoid misleading info

    # Log information about the machine (CPU, GPU, memory):
    run_results['machine_config'] = mutils.get_machine_config()

    # Let's measure the total time, including download of data
    start_time_global = datetime.datetime.now().strftime(cfg.TIME_FORMAT)

    # Locate training dataset
    # For real data, check whether the data was mounted to the right place
    # and if not, download it (imagenet_mini, cifar10, NOT imagenet!)
    if dataset_name == 'cifar10':
        mutils.locate_cifar10()
    if dataset_name == 'imagenet_mini':
        mutils.locate_imagenet_mini()
    if dataset_name == 'imagenet':
        mutils.locate_imagenet()

    if cfg.BENCHMARK_TYPE == 'pro':
        train_pro.train(train_args, kwargs, run_results)
    else:
        train_sd.train(kwargs, run_results)

    end_time_global = datetime.datetime.now().strftime(cfg.TIME_FORMAT)
    run_results['global_start_time'] = start_time_global
    run_results['global_end_time'] = end_time_global
    end_time_global = mutils.timestr_to_stamp(end_time_global, cfg.TIME_FORMAT)
    start_time_global = mutils.timestr_to_stamp(start_time_global, 
                                                cfg.TIME_FORMAT)
    run_results['global_execution_time_sec'] = (end_time_global - 
                                                start_time_global)

    return run_results


def get_train_args():
    """
    Returns a dict of dicts to feed the deepaas API parser
    """
    d_train = cfg.get_train_args_schema().fields
        
    if num_local_gpus == 0:
        d_train['num_gpus'] = fields.Int(missing=0,
                              description= 'Number of GPUs to train on \
                              (one node only). If set to zero, CPU is used.',
                              required= False
                              )
    else:
        num_gpus = [0]
        for i in range(num_local_gpus): num_gpus.append(str(i+1))
        d_train['num_gpus'] = fields.Int(missing=1,
                              description= 'Number of GPUs to train on \
                              (one node only). If set to zero, \
                              CPU is used.',
                              enum = num_gpus,
                              required= False
                              )

    # dictionary sorted by key, 
    # https://docs.python.org/3.6/library/collections.html#ordereddict-examples-and-recipes
    train_args = OrderedDict(sorted(d_train.items(), key=lambda t: t[0]))

    return train_args


# !!! deepaas>=1.0.0 calls get_predict_args() to get args for 'predict'
def get_predict_args():

    d_predict = cfg.PredictArgsSchema().fields
    # dictionary sorted by key, 
    # https://docs.python.org/3.6/library/collections.html#ordereddict-examples-and-recipes
    predict_args = OrderedDict(sorted(d_predict.items(), key=lambda t: t[0]))

    return predict_args

