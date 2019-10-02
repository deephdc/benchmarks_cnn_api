# -*- coding: utf-8 -*-
"""
Model description
"""

import argparse
import pkg_resources
import yaml
import json
# import project's config.py
import benchmarks_api.config as cfg

import benchmark_cnn as benchmark


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
            if line.startswith(par+":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta


def predict_file(*args):
    """
    Function to make prediction on a local file
    """
    message = 'Not implemented in the model (predict_file)'
    return message


def predict_data(*args):
    """
    Function to make prediction on an uploaded file
    """
    message = 'Not implemented in the model (predict_data)'
    return message


def predict_url(*args):
    """
    Function to make prediction on a URL
    """
    message = 'Not implemented in the model (predict_url)'
    return message


###
# Uncomment the following two lines
# if you allow only authorized people to do training
###
#import flaat
#@flaat.login_required()
def train(train_args):
    """
    Train network
    train_args : dict
        Json dict with the user's configuration parameters.
        Can be loaded with json.loads() or with yaml.safe_load()    
    """

    run_results = { "status": "ok",
                    "train_args": [],
                    "training": []
                  }
    run_results["train_args"].append(train_args)

    # Remove possible existing model files
    for f in os.listdir(cfg.MODEL_DIR):
        file_path = os.path.join(cfg.MODEL_DIR, f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    # TODO? Check for cpu/gpu version and adjust data format
             # 'data_format': 'NHWC',
    kwargs = {'model': yaml.safe_load(train_args.model),
              'num_gpus': yaml.safe_load(train_args.num_gpus),
	      'num_epochs': yaml.safe_load(train_args.num_epochs),
	      'batch_size': yaml.safe_load(train_args.batch_size),
	      'optimizer': yaml.safe_load(train_args.optimizer),
	      'local_parameter_device': 'cpu',
	      'variable_update': 'parameter_server',
	     # 'log_dir': cfg.DATA_DIR,
	      'benchmark_log_dir': cfg.DATA_DIR,
	      'train_dir': cfg.MODEL_DIR,
	      }

    if yaml.safe_load(train_args.dataset) != 'Synthetic data':
        kwargs['data_name'] = yaml.safe_load(train_args.dataset)
        # data_dir        

    params = benchmark.make_params(**kwargs)

    # TODO check wether paramers fit /catch
    params = benchmark.setup(params)
    bench = benchmark.BenchmarkCNN(params)

#    tfversion = cnn_util.tensorflow_version_tuple()
#    log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

    bench.print_info()
    bench.run()
		  
    # Read log file and get training results
    # training results: number of steps, avg img/sec
    logfile = '{}/metric.log'.format(cfg.DATA_DIR)
    with open(logfile, "r") as f:
        for line in f:
            pass
        result = json.loads(line)
        avg_examples = result["average_examples_per_sec"]
        run_results["training"]["average_examples_per_sec"] = avg_examples

    print(run_results)
    return run_results


def get_train_args():
    """
    Returns a dict of dicts to feed the deepaas API parser
    """
    train_args = cfg.train_args

    # convert default values and possible 'choices' into strings
    for key, val in train_args.items():
        val['default'] = str(val['default']) #yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]

    return train_args

# !!! deepaas>=0.5.0 calls get_test_args() to get args for 'predict'
def get_test_args():
    predict_args = cfg.predict_args

    # convert default values and possible 'choices' into strings
    for key, val in predict_args.items():
        val['default'] = str(val['default'])  # yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]

    return predict_args

# during development it might be practical 
# to check your code from the command line
def main():
    """
       Runs above-described functions depending on input parameters
       (see below an example)
    """

    if args.method == 'get_metadata':
        get_metadata()       
    elif args.method == 'train':
        train(args)
    else:
        get_metadata()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters')

    # get arguments configured for get_train_args()
    train_args = get_train_args()
    for key, val in train_args.items():
        parser.add_argument('--%s' % key,
                            default=val['default'],
                            type=type(val['default']),
                            help=val['help'])

    parser.add_argument('--method', type=str, default="get_metadata",
                        help='Method to use: get_metadata (default), \
                        predict_file, predict_data, predict_url, train')
    args = parser.parse_args()

    main()
