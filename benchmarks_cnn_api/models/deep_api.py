# -*- coding: utf-8 -*-
"""
Model description
"""

import argparse
import datetime
import json
import mimetypes
import os
import pkg_resources
import re
import shutil
import tarfile
import tempfile
import time
import urllib.request
import urllib.error

from tensorflow.python.client import device_lib
from werkzeug.exceptions import BadRequest
from webargs import fields
# import project's config.py
import benchmarks_cnn_api.config as cfg
import benchmark_cnn as benchmark
import cnn_util

from aiohttp.web import HTTPBadRequest
from collections import OrderedDict

## DEEPaaS wrapper to get e.g. UploadedFile() object
from deepaas.model.v2 import wrapper

## Authorization
from flaat import Flaat
flaat = Flaat()


# Switch for debugging in this script
debug_model = True

# Info on available GPUs
local_gpus = []
num_local_gpus = 0
# Available models for the data sets
models_cifar10 = ('alexnet', 'resnet56', 'resnet110')
models_imagenet = ('alexnet', 'resnet50', 'resnet152', 'mobilenet', 'vgg16', 
                   'vgg19', 'googlenet', 'overfeat', 'inception3')

time_fmt = '%Y-%m-%dT%H:%M:%S.%fZ'  # Timeformat of tf-benchmark

TMP_DIR = tempfile.gettempdir() # set the temporary directory


def _catch_error(f):
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
    
    print("predict(**kwargs) - kwargs: %s" % (kwargs)) if debug_model else ''

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
# Uncomment the following two lines
# if you allow only authorized people to do training
###
@flaat.login_required()
def train(**train_kwargs):
    """
    Train network
    train_args : dict
    """

    print("train(**train_kwargs) - train_kwargs: %s" % (train_kwargs)) if debug_model else ''

    # use the schema
    schema = cfg.TrainArgsSchema()
    # deserialize key-word arguments
    train_args = schema.load(train_kwargs)

    run_results = {"status": "ok", 
                   "user_args": train_args, 
                   "machine_config": {}, 
                   "training": {}, 
                   "evaluation": {}
                  }

    timestamp = int(datetime.datetime.timestamp(datetime.datetime.now()))
    Train_Run_Dir = os.path.join(cfg.MODELS_DIR, str(timestamp))

    if not os.path.exists(Train_Run_Dir):
        os.makedirs(Train_Run_Dir)
    else:
        raise BadRequest(
                "Directory to store training results, {}, already exists!"
                .format(Train_Run_Dir))
        
    # Remove possible existing model and log files
    #for f in os.listdir(Train_Run_Dir):
    #    file_path = os.path.join(Train_Run_Dir, f)
    #    try:
    #        if os.path.isfile(file_path):
    #            os.unlink(file_path)
    #    except Exception as e:
    #        print(e)

    # Declare training arguments
    kwargs = {'model': train_args['model'].split(' ')[0],
              'num_gpus': train_args['num_gpus'],
              'num_epochs': train_args['num_epochs'],
              'batch_size': train_args['batch_size_per_device'],
              'optimizer': train_args['optimizer'],
              'local_parameter_device': 'cpu',
              'variable_update': 'parameter_server'
              }

    # Locate training data and check if the selected network fits it
    # For real data check whether the right data was mounted to the right place and if not download it (cifar10 only)
    if train_args['dataset'] != 'Synthetic data':
        data_name = train_args['dataset']
        if data_name == 'cifar10':
            locate_cifar10()
        if data_name == 'imagenet':
            locate_imagenet()

        kwargs['data_name'] = data_name
        if data_name == 'imagenet_mini':
            locate_imagenet_mini()
            kwargs['data_name'] = 'imagenet'
        verify_selected_model(kwargs['model'], kwargs['data_name'])
        kwargs['data_dir'] = '{}/{}'.format(cfg.DATA_DIR, data_name)
    else:
        verify_selected_model(kwargs['model'], 'imagenet')

    # If no GPU is available or the gpu option is set to 0 run CPU mode
    if num_local_gpus == 0 or kwargs['num_gpus'] == 0:
        kwargs['device'] = 'cpu'
        kwargs['data_format'] = 'NHWC'  # cpu data format
        kwargs['num_gpus'] = 1  # Important: tensorflow uses this also to specify the number of CPUs
    else:
        kwargs['device'] = 'gpu'
        kwargs['data_format'] = 'NCHW'

    # Add training info to run_results but not the directories
    run_results["training"].update(kwargs)
    if run_results["training"]["device"] == "cpu":
        del run_results["training"]["num_gpus"]  # avoid misleading info
    kwargs['train_dir'] = Train_Run_Dir
    kwargs['benchmark_log_dir'] = Train_Run_Dir


    # Setup and run the benchmark model
    params = benchmark.make_params(**kwargs)
    try:
        params = benchmark.setup(params)
        bench = benchmark.BenchmarkCNN(params)
    except ValueError as param_ex:
        raise BadRequest("ValueError in parameter setup: {}. Params: {}".format(param_ex, params))

    tf_version = '.'.join([str(x) for x in cnn_util.tensorflow_version_tuple()])
    run_results["training"]["tf_version"] = tf_version

    # Run benchmark and measure total execution time
    bench.print_info()
    start_time_global = datetime.datetime.now().strftime(time_fmt)
    try:
        bench.run()
    except ValueError as ve:
        raise BadRequest('ValueError in benchmark execution: {}'.format(ve))
    end_time_global = datetime.datetime.now().strftime(time_fmt)

    # Read training and metric log files and store training results
    training_file = '{}/training.log'.format(Train_Run_Dir)
    os.rename('{}/benchmark_run.log'.format(Train_Run_Dir), training_file)
    run_parameters, machine_config = parse_logfile_training(training_file)
    run_results['training'].update(run_parameters)
    run_results["machine_config"] = machine_config

    metric_file = '{}/metric.log'.format(Train_Run_Dir)
    run_results['training']['result'] = {}
    run_results['training']['result']['global_start_time'] = start_time_global
    run_results['training']['result']['global_end_time'] = end_time_global
    start, end, avg_examples = parse_metric_file(metric_file)
    run_results["training"]["result"]["average_examples_per_sec"] = avg_examples
    run_results['training']['result']['execution_start_time'] = start
    run_results['training']['result']['execution_end_time'] = end


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
                       # 'eval_dir': cfg.DATA_DIR,
                       }
        run_results['evaluation']['device'] = kwargs_eval['device']
        if run_results['evaluation']['device'] == 'gpu':
            run_results['evaluation']['num_gpus'] = kwargs_eval['num_gpus']  # only for GPU to avoid confusion

        # Locate data
        if train_args['dataset'] != 'Synthetic data':
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
        start_time_global = datetime.datetime.now().strftime(time_fmt)
        evaluation.run()
        end_time_global = datetime.datetime.now().strftime(time_fmt)


        # Read log files and get evaluation results
        os.rename('{}/benchmark_run.log'.format(Train_Run_Dir), '{}/evaluation.log'.format(Train_Run_Dir))
        evaluation_file = '{}/evaluation.log'.format(Train_Run_Dir)
        run_parameters = parse_logfile_evaluation(evaluation_file)
        run_results['evaluation'].update(run_parameters)

        logfile = '{}/metric.log'.format(Train_Run_Dir)
        run_results['evaluation']['result'] = {}
        run_results['evaluation']['result']['global_start_time'] = start_time_global
        run_results['evaluation']['result']['global_end_time'] = end_time_global

        with open(logfile, "r") as f:
            for line in f:
                l = json.loads(line)
                if l["name"] == "eval_average_examples_per_sec":
                    run_results["evaluation"]['result']["average_examples_per_sec"] = l["value"]
                if l["name"] == "eval_top_1_accuracy":
                    run_results["evaluation"]['result']["top_1_accuracy"] = l["value"]
                if l["name"] == "eval_top_5_accuracy":
                    run_results["evaluation"]['result']["top_5_accuracy"] = l["value"]

    return run_results


def download_untar_public(dataset, remote_url, tar_mode="r"):
    """
    Download dataset from the public URL and untar
    """
    dataset_dir = os.path.join(cfg.DATA_DIR, dataset)    

    #url, filename = os.path.split(remote_url)
    tmp_dataset = os.path.join(TMP_DIR, dataset+".tmp")
    try:
        fileName, header = urllib.request.urlretrieve(remote_url,
                                                      filename=tmp_dataset)
        print('[INFO] Extracting tar-archive...')
        with tarfile.open(name=fileName, mode=tar_mode) as tar:
            # archive name and dataset name maybe different
            # de-archive, then move files one-by-one to dataset_dir
            tar.extractall(path=TMP_DIR)
            rootdir = os.path.commonpath(tar.getnames())
            rootdir = os.path.join(TMP_DIR, rootdir)
            for f in os.listdir(rootdir):
                # if some files already exist, delete them and re-copy
                try:
                    shutil.move(os.path.join(rootdir, f), dataset_dir)
                except OSError:
                    msg = '[WARNING] {} probably found in {}, '.format(f, dataset_dir) + \
                    "trying to remove it and re-copy.."
                    print(msg)
                    os.remove(os.path.join(dataset_dir, f))
                    shutil.move(os.path.join(rootdir, f), dataset_dir)
                
            shutil.rmtree(rootdir) # 'strong' remove of the directory, i.e. if not empty
            os.remove(tmp_dataset)
        print(('[INFO] Done extracting files to {}'.format(dataset_dir)))

    except urllib.error.HTTPError as e:
        raise BadRequest('[ERROR] No local dataset found at {}.\
        But also could not retrieve data from "{}"!'.format(dataset_dir, 
                                                            remote_url))

def locate_cifar10():
    """
     Check if the necessary Cifar10 files are available locally in the 'data' directory.
     If not, download them from the official page and extract
    """
    # Files of the Cifar10 Dataset
    cifar10_files = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    cifar10Local = True
    cifar10_dir = os.path.join(cfg.DATA_DIR, 'cifar10')

    # Check local availability
    if not os.path.exists(cifar10_dir):
        os.makedirs(cifar10_dir)
        cifar10Local = False
    else:
        for f in cifar10_files:
            if not os.path.exists(os.path.join(cifar10_dir, f)):
                cifar10Local = False

    # If not available locally, download to data directory
    if not cifar10Local:
        print(('[WARNING] No local copy of Cifar10 found.\
        Trying to download from {}'.format(cfg.CIFAR10_REMOTE_URL)))
        download_untar_public('cifar10', cfg.CIFAR10_REMOTE_URL, 'r:gz')


def locate_imagenet_mini():
    """
    Check if ImageNet (mini) is in the required folder
    """
    imagenet_mini_dir = os.path.join(cfg.DATA_DIR, 'imagenet_mini')

    # Check local availability
    if not os.path.exists(imagenet_mini_dir):
        os.makedirs(imagenet_mini_dir)
        print(('[WARNING] No local copy of imagenet_mini found. \
        Trying to download from {}'.format(cfg.IMAGENET_MINI_REMOTE_URL)))
        download_untar_public('imagenet_mini', cfg.IMAGENET_MINI_REMOTE_URL)


def locate_imagenet():
    """
    Check if ImageNet is in the required folder
    """
    imagenet_dir = os.path.join(cfg.DATA_DIR, 'imagenet')
    if not os.path.exists(imagenet_dir):
        raise BadRequest('No local ImageNet dataset found at {}!'.format(imagenet_dir))


def verify_selected_model(model, data_set):
    """
    Check if the user has selected a model that is compatible with the data set
    """
    if data_set == 'cifar10':
        if model not in models_cifar10:
            raise BadRequest('Unsupported model selected! Cifar10 dataset supported models are: {}'
                         .format(models_cifar10))
    if data_set == 'imagenet':
        if model not in models_imagenet:
            raise BadRequest('Unsupported model selected! ImageNet dataset supported models are: {}'
                             .format(models_imagenet))


def parse_logfile_training(logFile):
    """ takes log file with benchmark settings in JSON format
        and parses relevant parts
    """
    run_parameters = {}
    with open(logFile, "r") as read_file:
        json_data = json.load(read_file)  # dictionary

        for el in json_data['run_parameters']:
            if el['name'] == 'batch_size':
                run_parameters['batch_size'] = el['long_value']
            if el['name'] == 'batch_size_per_device':
                try:
                    run_parameters['batch_size_per_device'] = el['float_value']
                except:
                    run_parameters['batch_size_per_device'] = el['long_value']
            if el['name'] == 'num_batches':
                run_parameters['num_batches'] = el['long_value']

        machine_config = json_data["machine_config"]

    return run_parameters, machine_config


def parse_logfile_evaluation(logFile):
    """ takes log file with evaluation settings in JSON format
        and parses relevant parts
    """
    run_parameters = {}
    with open(logFile, "r") as read_file:
        json_data = json.load(read_file)  # dictionary

        for el in json_data['run_parameters']:
            if el['name'] == 'batch_size':
                run_parameters['batch_size'] = el['long_value']
            if el['name'] == 'batch_size_per_device':
                try:
                    run_parameters['batch_size_per_device'] = el['float_value']
                except:
                    run_parameters['batch_size_per_device'] = el['long_value']
            if el['name'] == 'num_batches':
                run_parameters['num_batches'] = el['long_value']
            if el['name'] == 'data_format':
                run_parameters['data_format'] = el['string_value']
            if el['name'] == 'model':
                run_parameters['model'] = el['string_value']
            # not sure why evaluation uses optimizer
            #if el['name'] == 'optimizer':
            #    run_parameters['optimizer'] = el['string_value']

    return run_parameters


def parse_metric_file(metric_file):
    """ takes the metric file and extracts timestamps and avg_imgs / sec info
    """
    with open(metric_file, "r") as f:
        maxStep, minTime, maxTime, avg_examples = 0, 0, 0, 0
        for line in f:
            el = json.loads(line)
            if el['name'] == "current_examples_per_sec" and el['global_step'] == 1:
                minTime = el['timestamp']
            if el['name'] == "current_examples_per_sec" and el['global_step'] > maxStep:
                maxTime = el['timestamp']
                maxStep = el['global_step']
            if el['name'] == 'average_examples_per_sec':
                avg_examples = el['value']
    return minTime, maxTime, avg_examples


def get_train_args():
    """
    Returns a dict of dicts to feed the deepaas API parser
    """
    d_train = cfg.TrainArgsSchema().fields
    global local_gpus, num_local_gpus

    # Adjust num_gpu option accordingly to available local devices
    local_devices = device_lib.list_local_devices()
    local_gpus = [x for x in local_devices if x.device_type == 'GPU']
    num_local_gpus = len(local_gpus)
        
    if num_local_gpus == 0:
        d_train['num_gpus'] = fields.Str(missing=0,
                              description= 'Number of GPUs to train on \
                              (one node only). If set to zero, CPU is used.',
                              required= False
                              )
    else:
        num_gpus = []
        for i in range(num_local_gpus): num_gpus.append(str(i+1))
        d_train['num_gpus'] = fields.Str(missing=1,
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


# during development it might be practical 
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """

    if args.method == 'get_metadata':
        meta = get_metadata()
        print(json.dumps(meta))
        return meta 
    elif args.method == 'predict':
        ## use the schema
        #schema = cfg.PredictArgsSchema()
        #result = schema.load(vars(args))
    
        # TODO: change to many files ('for' itteration)
        if args.files:
            # create tmp file as later it will be deleted
            temp = tempfile.NamedTemporaryFile()
            temp.close()
            # copy original file into tmp file
            with open(args.files, "rb") as f:
                with open(temp.name, "wb") as f_tmp:
                    for line in f:
                        f_tmp.write(line)
        
            # create file object to mimic aiohttp workflow
            file_obj = wrapper.UploadedFile(name="data", 
                                            filename = temp.name,
                                            content_type=mimetypes.MimeTypes().guess_type(args.files)[0],
                                            original_filename=args.files)
            args.files = file_obj
        
        results = predict(**vars(args))
        print(json.dumps(results))
        return results        
    elif args.method == 'train':
        start = time.time()
        results = train(**vars(args))
        print("Elapsed time:  ", time.time() - start)
        print(json.dumps(results))
        return results


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Model parameters', 
                                     add_help=False)
    
    cmd_parser = argparse.ArgumentParser()    
    subparsers = cmd_parser.add_subparsers(
                            help='methods. Use \"model.py method --help\" to get more info', 
                            dest='method')

    get_metadata_parser = subparsers.add_parser('get_metadata', 
                                         help='get_metadata method',
                                         parents=[parser])

    # get train arguments configured
    train_parser = subparsers.add_parser('train', 
                                         help='commands for training',
                                         parents=[parser])
    train_args = _fields_to_dict(get_train_args())
    for key, val in train_args.items():
        train_parser.add_argument('--%s' % key,
                               default=val['default'],
                               type=val['type'], #may just put str
                               help=val['help'],
                               required=val['required'])

    # get predict arguments configured
    predict_parser = subparsers.add_parser('predict', 
                                           help='commands for prediction',
                                           parents=[parser])

    predict_args = _fields_to_dict(get_predict_args())
    for key, val in predict_args.items():
        predict_parser.add_argument('--%s' % key,
                               default=val['default'],
                               type=val['type'], #may just put str
                               help=val['help'],
                               required=val['required'])

    args = cmd_parser.parse_args()
   
    main()

