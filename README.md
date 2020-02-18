DEEP Open Catalogue: API for Tensorflow Benchmarks
==============================

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/benchmarks_cnn_api/test)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/benchmarks_cnn_api/job/test)

[tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) from TensorFlow team accessed via [DEEPaaS API](https://github.com/indigo-dc/DEEPaaS), currently V1.

This is a **wrapper** to access the [TF Benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks), **not the benchmarks code** itself! 
One has to install [tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks)
and [TensorFlow Models](https://github.com/tensorflow/models) and make them accessible in Python. For example for TF 1.10.0, as

```bash
$ git clone --depth 1 -b cnn_tf_v1.10_compatible https://github.com/tensorflow/benchmarks.git
$ export PYTHONPATH=$PYTHONPATH:$PWD/benchmarks/scripts/tf_cnn_benchmarks
```

The **recommended way** to run [TF Benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) through [DEEPaaS API](https://github.com/indigo-dc/DEEPaaS)
is to use our [Docker images](https://hub.docker.com/r/deephdc/deep-oc-benchmarks_cnn) also available through the [DEEP Open Catalog](https://marketplace.deep-hybrid-datacloud.eu/modules/deep-oc-benchmarks-cnn.html).
The Docker images already **contain** [TF Benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) and [DEEPaaS API](https://github.com/indigo-dc/DEEPaaS).

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
    │   │   └── deep_api.py
    │   │
    │   └── tests              <- Scripts to perfrom code testing + pylint script
    │   │
    │   └── visualization      <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini                <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


