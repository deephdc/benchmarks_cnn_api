import benchmarks_cnn_api.config as cfg
import datetime
import json
import logging
import multiprocessing
import os
import progressbar
import shutil
import tarfile
import tempfile
import tensorflow as tf
import time
import urllib.request
import urllib.error

from tensorflow.python.client import device_lib
from werkzeug.exceptions import BadRequest

# conigure python logger
logger = logging.getLogger('__name__') #o3api
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s')
logger.setLevel(cfg.log_level)

TMP_DIR = tempfile.gettempdir() # set the temporary directory

# Available models for the data sets
models_cifar10 = ('alexnet', 'resnet56', 'resnet110')
models_imagenet = ('alexnet', 'resnet50', 'resnet152', 'mobilenet', 'vgg16', 
                   'vgg19', 'googlenet', 'overfeat', 'inception3', 'trivial')

progress_bar = None

def download_untar_public(dataset, remote_url, tar_mode="r"):
    """
    Download dataset from the public URL and untar
    """
    dataset_dir = os.path.join(cfg.DATA_DIR, dataset)    

    #url, filename = os.path.split(remote_url)
    tmp_dataset = os.path.join(TMP_DIR, dataset+".tmp")

    # https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
    def _progress(block_num, block_size, total_size):
        global progress_bar
        if progress_bar is None:
            progress_bar = progressbar.ProgressBar(maxval=total_size)
            progress_bar.start()
    
        downloaded = block_num * block_size
        if downloaded < total_size:
            progress_bar.update(downloaded)
        else:
            progress_bar.finish()
            progress_bar = None

    try:
        logger.info('[INFO] Downloading in {}'.format(tmp_dataset))
        fileName, header = urllib.request.urlretrieve(remote_url,
                                                      tmp_dataset,
                                                      _progress)
        logger.info('[INFO] Extracting tar-archive...')
        with tarfile.open(name=fileName, mode=tar_mode) as tar:
            # archive name and dataset name maybe different
            # de-archive, then move files one-by-one to dataset_dir
            tar.extractall(path=TMP_DIR)
            rootdir = os.path.commonpath(tar.getnames())
            rootdir = os.path.join(TMP_DIR, rootdir)
            for f in os.listdir(rootdir):
                # if some files already exist, delete them and re-copy
                try:
                    if not os.path.exists(dataset_dir):
                        os.makedirs(dataset_dir)
                    shutil.move(os.path.join(rootdir, f), dataset_dir)
                except OSError:
                    msg = '[WARNING] {} probably found in {}, '.format(f, dataset_dir) + \
                    "trying to remove it and re-copy.."
                    logger.warning(msg)
                    os.remove(os.path.join(dataset_dir, f))
                    shutil.move(os.path.join(rootdir, f), dataset_dir)
                
            shutil.rmtree(rootdir) # 'strong' remove of the directory, i.e. if not empty
            os.remove(tmp_dataset)
        logger.info(('[INFO] Done extracting files to {}'.format(dataset_dir)))

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
        logger.warning(('[WARNING] No local copy of Cifar10 found.\
        Trying to download from {}'.format(cfg.CIFAR10_REMOTE_URL)))
        download_untar_public('cifar10', cfg.CIFAR10_REMOTE_URL, 'r:gz')


def locate_imagenet_mini():
    """
    Check if ImageNet (mini) is in the required folder
    """
    imagenet_mini_dir = os.path.join(cfg.DATA_DIR, 'imagenet_mini')

    # Check local availability
    if not os.path.exists(imagenet_mini_dir):
        #os.makedirs(imagenet_mini_dir)
        logger.warning(('[WARNING] No local copy of imagenet_mini found. \
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
            raise BadRequest('Unsupported model selected, {}! \
            Cifar10 dataset supported models are: {}'
            .format(model, models_cifar10))

    if data_set == 'imagenet':
        if model not in models_imagenet:
            raise BadRequest('Unsupported model selected, {}! \
            ImageNet dataset supported models are: {}'
            .format(model, models_imagenet))

def create_train_run_dir(kwargs):
    """Function to create Train_Run_Dir to store training data
    """
    timestamp = int(datetime.datetime.timestamp(datetime.datetime.now()))
    Train_Run_Dir = os.path.join(cfg.MODELS_DIR, str(timestamp))
    Eval_Dir = os.path.join(Train_Run_Dir, 'eval_dir')

    if not os.path.exists(Train_Run_Dir):
        try:
            os.makedirs(Train_Run_Dir)
        except OSError as e:
           logger.warning('OSError: {}. Directory {} seems to exist'
                           .format(e, Train_Run_Dir))
           pass


    if not os.path.exists(Eval_Dir):
        try:
            os.makedirs(Eval_Dir)
        except OSError as e:
            logger.warning('OSError: {}. Directory {} seems to exist'
                           .format(e, Eval_Dir))
            pass
    #else:
    #    # The following fails for horovod (or other parallelisation)
    #    raise BadRequest(
    #            "Directory to store training results, {}, already exists!"
    #            .format(Train_Run_Dir))    

    return Train_Run_Dir, Eval_Dir

def _collect_cpu_info(machine_info):
    """Collect the CPU information for the local environment.
    Copy from tensorflow/models/officials/utils/logs/logger.py
    """
    cpu_info = {}
    cpu_info["num_cores"] = multiprocessing.cpu_count()

    try:
    # Note: cpuinfo is not installed in the TensorFlow OSS tree.
    # It is installable via pip.
        import cpuinfo    # pylint: disable=g-import-not-at-top

        info = cpuinfo.get_cpu_info()

        try:
            cpu_info["cpu_info"] = info["brand"]
        except:
            # py-cpuinfo >v5.0.0
            cpu_info["cpu_info"] = info["brand_raw"]

        try:
            cpu_info["mhz_per_cpu"] = info["hz_advertised_raw"][0] / 1.0e6
        except:
            # py-cpuinfo >v5.0.0
            cpu_info["mhz_per_cpu"] = info["hz_advertised"][0] / 1.0e6

        machine_info["cpu_info"] = cpu_info
    except ImportError:
        tf.logging.warn("'cpuinfo' not imported. CPU info will not be logged.")


def _collect_memory_info(machine_info):
    """Collect Memory information for the local environment.
    Copy from tensorflow/models/officials/utils/logs/logger.py
    """    
    try:
        # Note: psutil is not installed in the TensorFlow OSS tree.
        # It is installable via pip.
        import psutil   # pylint: disable=g-import-not-at-top
        vmem = psutil.virtual_memory()
        machine_info["memory_total"] = vmem.total
        machine_info["memory_available"] = vmem.available
    except ImportError:
        tf.logging.warn("'psutil' not imported. Memory info will not be logged.")

def _parse_gpu_model(physical_device_desc):
    # Assume all the GPU connected are same model
    for kv in physical_device_desc.split(","):
        k, _, v = kv.partition(":")
        if k.strip() == "name":
            return v.strip()
    return None

def get_available_gpus():
    """Function to get number of local GPUs and their Memory size
    according to available local devices.
    Inspired by tensorflow/models/officials/utils/logs/logger.py
    """
    local_devices = device_lib.list_local_devices()
    local_gpus = [x for x in local_devices if x.device_type == 'GPU']
    num_local_gpus = len(local_gpus)
    # Assume all the GPU connected are same model
    if num_local_gpus > 0:
        gpu_model = _parse_gpu_model(local_gpus[0].physical_device_desc)
        gpu_memory_limit = local_gpus[0].memory_limit
    else:
        gpu_model = "No GPU found"
        gpu_memory_limit = 0

    return num_local_gpus, gpu_model, gpu_memory_limit

def get_machine_config():
    """Function to retrive machine config. We re-implement this function but
    to a large extend this is a copy from 
    tensorflow/models/officials/utils/logs/logger.py
    """
    machine_config = { 'cpu_info': {},
                       'gpu_info': {}
                     }
    _collect_cpu_info(machine_config)

    num_gpus, gpu_model, gpu_memory = get_available_gpus()
    machine_config['gpu_info']['count'] = num_gpus
    machine_config['gpu_info']['model'] = gpu_model
    machine_config['gpu_info']['memory'] = gpu_memory
    
    _collect_memory_info(machine_config)
    
    return machine_config

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

    return run_parameters


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

    maxStep, minTime, maxTime, avg_examples = 0, 0, 0, 0
    with open(metric_file, "r") as f:
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


def timestr_to_stamp(time_str, time_format):
    """Function to convert time string to timestamp
    
    :par time_str: string representing time
    """
    time = datetime.datetime.strptime(time_str, time_format)
    timestamp = datetime.datetime.timestamp(time)
    return timestamp


def wait_final_read(input_file, parameter):
    """Function to ensure successful read of the parameter"""
    max_cycles = 25 # to avoid an infinite loop
    icyc = 0
    read = False
    print("[DEBUG] read {} is"
          .format(parameter), end = ' ') if cfg.DEBUG_MODEL else ''
    while icyc < max_cycles and not read: 
        with open(input_file, "r") as f:
            for line in f:
                l = json.loads(line)
                if l["name"] == parameter:
                    read = True
                    
        print("{}"
              .format(read), end = ' ') if cfg.DEBUG_MODEL else ''
        icyc += 1
        time.sleep(1)
    
    print("") if cfg.DEBUG_MODEL else ''