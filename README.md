# VDiscover 

VDiscover is a tool designed to train a vulnerability detection predictor.
Given a vulnerability discovery procedure and a **large** enough number of training testcases, it extracts **lightweight** features to predict which testcases are potentially vulnerable. This repository contains an improved version of a proof-of-concept used to show experimental results in our technical report (available [here](http://vdiscover.org/report.pdf)).

## Use cases

VDiscover aims to be used when there is a **large** amount of testcases to analyze using a **costly** vulnerability detection procedure. It can be trained to provide a quick prioritization of testcases. The extraction of features to perform a prediction is designed to be scalable. Nevertheless, this implementation is not particularly optimized so it should easy to improve the performance of it.

## Requirements

* Python 2.7
* binutils
* [python-ptrace](https://bitbucket.org/haypo/python-ptrace/)
* [scikit-learn](http://scikit-learn.org/) for training/testing
* [matplotlib](http://matplotlib.org/) for visualization (very experimental, optional)
* [keras](http://keras.io/) for convolutional clustering (very experimental, optional)


## Quickstart

Before starting, it is recommended to manually install binutils, scikit-learn and setuptools (to perform a local installation). Then we can execute:

    git clone https://github.com/CIFASIS/VDiscover.git
    cd VDiscover
    python setup.py install --user

By default, the local installation of the command line utilities of VDiscover is performed inside ~/.local/bin, so it is recommended to add this directory into the PATH variable. Our tool is composed by two main components:

* **fextractor**: to extract dynamic and static features from test cases.
* **vpredictor**: to train a new vulnerability prediction model or predict using a previously trained one. It can be used to cluster and visualize a set of test cases.

Some examples of testcases of very popular programs (grep, gzip, bc, ..) can be found in  [examples/testcases](examples/testcases).  For example, to extract raw dynamic features from an execution of [bc](http://www.gnu.org/software/bc/):

    fextractor --dynamic bc 

And the resulted extracted features are:

    /usr/bin/bc	isatty:0=Num32B0 isatty:0=Num32B8 setvbuf:0=Ptr32 setvbuf:1=NPtr32 setvbuf:2=Num32B8 setvbuf:3=Num32B0 ...

This raw data can be used to train a new vulnerability prediction model or predict using a previously trained one. Additionally, more detailed (but outdated) documentation is available [here](doc/index.md).

## License

[GPL3](LICENSE)
