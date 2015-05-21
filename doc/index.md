# Technical Documentation of VDiscover

## Introduction

*VDiscover* is a Python tool that extracts lightweight features to predict which testcases are potentially vulnerable, given a large enough number of training examples.  The code contained in this repository is an improved version of a proof-of-concept used to show experimental results in our technical report. This document explains concrete technical aspects of the VDiscover implementation but in order to understand the concepts behind this technique, you should consult the [technical report](http://www.vdiscover.org/report.pdf).

## Supported Arquitectures

### x86
* IA-32

    Fully supported to extract dynamic and static features. All the experiments performed in our publication used the IA-32 architecture.

* [x32](https://lwn.net/Articles/456731/)

    Almost completely supported to extract dynamic and static features. Only the detection of memory corruption was disabled, but we need to try with more testcases to be sure. We are currently limited by the [lack of support of python-ptrace](https://bitbucket.org/haypo/python-ptrace/issue/12/feature-request-support-x86-arch-on-x86_64). Nevertheless, the feature extraction should work correctly.

* x86-64

    This architecture is partially supported. Only extraction of static features is available (since objdump will take care of the dissasembly of ELF files). The implementation of the dynamic feature extraction needs to deal with its calling convention and it not implemeted yet. 

## Feature extractor

**fextractor** is a Python script to perform static and dynamic feature extraction from a testcase. To know more about features, you can consult either the [technical report](http://www.vdiscover.org/report.pdf) or the source code of VDiscover to understand exactly how they are extracted.

### Static features

Static features are supposed to capture information relevant to a whole program, and they should be obtained without running the code on particular inputs. The lightweight approach we implemented randomly walks the approximate control flow graph of the binary collecting sequences of potential C standard library calls. Different parameters like *--max-subtraces-collected* and *--max-subtraces-explored* are defined to control the number of call sequences sampled from the control flow graph.

### Dynamic features

Dynamic features are supposed to capture a sample of the behavior of a program in terms of its concrete sequential calls to the C standard library. Additionally the final state of the execution is included. Such features are extracted by executing for a limited time a testcase and hooking program
events, collecting them in a sequence. Since an execution can involve several modules, we included some command line options to include or ignore them during the extraction process.

#### Including or ignoring modules

By default, dynamic features will be extracted analyzing all the libraries used by a program. Nevertheless, it is possible to include or ignore modules using *--inc-mods* and *--ign-mods*. Modules should be listed in a file using different lines. Every line will be matched against all the linked library in a file. For example, to include or ignore *libjpeg.so.8*, it should be enough to add the following line:

    libjpeg.so

Dynamically loaded libraries (e.g. dlopen) are not supported (but it should be relatively easy to implement).

## Vulnerability Predictor

**vpredictor** is Python script to train a new vulnerability prediction model or predict using a previously trained one. 

In order to use this utility  either for training or prediction, some input data should be provided in csv format (delimited by "\t"). Fortunately, **fextractor** automatically outputs in such format.

By default, it works in prediction mode. Therefore, a previously trained model is mandatory and should be specified using the **--model** command line option.


# Tutorials

This section contains a few simple tutorials to use **fextractor** and **vpredictor**.

## Creating new testcases

A testcase is a directory packing all the inputs necessary to reproduce a particular behavior of a executable file. A typical testcase folder has the following structure:

    program      
      path.txt
      inputs
        argv_1.symb
        argv_2.symb
        ...
        file_filename1.ext.symb
        file_filename2.ext.symb
        ...

The *path.txt* file just contains the complete absolute path of a binary program to be analyzed.
The *inputs* folder contains one file for every input used to reproduce a testcase. Arguments are specified using *argv_N.symb* files where N is the position of each one. Normal files required for a program to run are named *file_filename1.ext.symb* where filename1.ext is the real filename. Every time the testcase is executed, these files are going to be re-created (just in case any of them was modified).
The special file *file___dev__stdin.symb* can be defined to provide standard input to a testcase. Other input sources, for instance environment variables, are not supported, but the code can easily be extended to do it.

The input folder is optional. If there is no *inputs*, the testcase will only contains the path to an executable file and therefore, it can only be used to extract static features. 

## Training a Predictor

A basic training support is included in **vpredictor**. After collecting enough train data (probably using **fextractor**), this script can be used to train a model, specifying which types of features should be used (static or dynamic) as well as the out file to dump the resulting model.

    vpredictor dataset.csv --train --dynamic --out-file dyn-predictor.pklz

The data provided should be properly balanced and shuffled. Gzipped csv are supported (automatically detected using the ".csv.gz" extension).
Training a Machine Learning model usually requires trying different options and parameters. In our prototype, we only provide the most robust classifier and parameters according to our experiments. This is not necessarily the best option to train with other types of vulnerabilities or testcases. For more information, consult the [scikit-learn documentation website](http://scikit-learn.org/stable/documentation.html).
After training, the same script can output the predictions for some new testcases:

    vpredictor test.csv --dynamic --model dyn-predictor.pklz --out-file predictions.csv
