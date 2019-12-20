DEEP Open Catalogue: API for Tensorflow Benchmarks
==============================

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/benchmarks_api/test)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/benchmarks_api/job/test)

[tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) from TensorFlow team accessed via [DEEPaaS API](https://github.com/indigo-dc/DEEPaaS), currently V1.

## From [tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks):
tf_cnn_benchmarks contains implementations of several popular convolutional
models, and is designed to be as fast as possible. tf_cnn_benchmarks supports
both running on a single machine or running in distributed mode across multiple
hosts. See the [High-Performance models
guide](https://www.tensorflow.org/performance/performance_models) for more
information.

These models utilize many of the strategies in the [TensorFlow Performance
Guide](https://www.tensorflow.org/performance/performance_guide). Benchmark
results can be found [here](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/performance/benchmarks.md).

These models are designed for performance. For models that have clean and
easy-to-read implementations, see the [TensorFlow Official
Models](https://github.com/tensorflow/models/tree/master/official).


## Project Organization

    ├── LICENSE
    ├── README.md              <- The top-level README for developers using this project.
    ├── data
    │   └── raw                <- The original, immutable data dump.
    │
    ├── docs                   <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                 <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                             the creator's initials (if many user development), 
    │                             and a short `_` delimited description, e.g.
    │                             `1.0-jqp-initial_data_exploration.ipynb`.
    │
    ├── references             <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures            <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
    │                             generated with `pip freeze > requirements.txt`
    ├── test-requirements.txt  <- The requirements file for the test environment
    │
    ├── setup.py               <- makes project pip installable (pip install -e .) so benchmarks_api can be imported
    ├── benchmarks_cnn_api    <- Source code for use in this project.
    │   ├── __init__.py        <- Makes benchmarks_api a Python module
    │   │
    │   ├── dataset            <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features           <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models             <- Scripts to train models and then use trained models to make
    │   │   │                     predictions
    │   │   └── deepaas_api.py
    │   │
    │   └── tests              <- Scripts to perfrom code testing + pylint script
    │   │
    │   └── visualization      <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini                <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Getting Started
To run ResNet50 with synthetic data without distortions with a single GPU, run

```
python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --variable_update=parameter_server
```

Note that the master branch of tf_cnn_benchmarks requires the latest nightly
version of TensorFlow. You can install the nightly version by running `pip
install tf-nightly-gpu` in a clean environment, or by installing TensorFlow from
source. We sometimes will create a branch of tf_cnn_benchmarks, in the form of
cnn_tf_vX.Y_compatible, that is compatible with TensorFlow version X.Y For
example, branch
[cnn_tf_v1.9_compatible](https://github.com/tensorflow/benchmarks/tree/cnn_tf_v1.9_compatible/scripts/tf_cnn_benchmarks)
works with TensorFlow 1.9.

Some important flags are

*   model: Model to use, e.g. resnet50, inception3, vgg16, and alexnet.
*   num_gpus: Number of GPUs to use.
*   data_dir: Path to data to process. If not set, synthetic data is used. 
*   batch_size: Batch size for each GPU.
*   variable_update: The method for managing variables: parameter_server
    ,replicated, distributed_replicated, independent
*   local_parameter_device: Device to use as parameter server: cpu or gpu.

To see the full list of flags, run `python tf_cnn_benchmarks.py --help`, or read [OPTIONS.txt](OPTIONS.txt)

To run ResNet50 with real data with 8 GPUs, run:

```
python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=256 \
--model=resnet50 --optimizer=momentum --variable_update=replicated \
--nodistortions --gradient_repacking=8 --num_gpus=8 \
--num_epochs=90 --weight_decay=1e-4 --data_dir=${DATA_DIR} --use_fp16 \
--train_dir=${CKPT_DIR}
```
This will train a ResNet-50 model on ImageNet with 2048 batch size on 8
GPUs. The model should train to around 76% accuracy.

## Using Real datasets

Four real datasets can be used:
*   CIFAR10: 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images [[CIFAR page from Uni Toronto]](https://www.cs.toronto.edu/~kriz/cifar.html)
*   IMAGENET: classical dataset features a collection of 1.2 million labeled images with one thousand object categories used for training data in the image-net competition [[Official webpage]](http://www.image-net.org/)
*   COCO: a large-scale object detection, segmentation, and captioning dataset [[Official webpage]](http://cocodataset.org/#home)
*   LIBRISPEECH: Large-scale (1000 hours) corpus of read English speech [[Official webpage]](http://www.openslr.org/12/)

#### CIFAR10 ####
Please, go to [[CIFAR page from Uni Toronto]](https://www.cs.toronto.edu/~kriz/cifar.html) and download the [dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz),
de-archive it in a certain directory, provide this directory with ```--data_dir``` flag. 

Optionally provide also ```--data_name=cifar10```.


#### IMAGENET #####
According the official [Tensorflow Benchmark README for TF1.10](https://github.com/tensorflow/benchmarks/tree/cnn_tf_v1.10_compatible/scripts/tf_cnn_benchmarks), 
in order to use Imagenet data use these
[instructions](https://github.com/tensorflow/models/tree/master/research/inception#getting-started)
as a starting point.


#### COCO #####
not yet tested by us

#### LIBRISPEECH ####
not yet tested by us

## Running the tests

To run the tests, run

```bash
pip install portpicker
python run_tests.py && python run_tests.py --run_distributed_tests
```

Note the tests require portpicker.

The command above runs a subset of tests that is both fast and fairly
comprehensive. Alternatively, all the tests can be run, but this will take a
long time:

```bash
python run_tests.py --full_tests && python run_tests.py --full_tests --run_distributed_tests
```

We will run all tests on every PR before merging them, so it is not necessary
to pass `--full_tests` when running tests yourself.

To run an individual test, such as method `testParameterServer` of test class
`TfCnnBenchmarksTest` of module `benchmark_cnn_test`, run

```bash
python -m unittest -v benchmark_cnn_test.TfCnnBenchmarksTest.testParameterServer
```
