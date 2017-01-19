"""
This file is part of VDISCOVER.

VDISCOVER is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

VDISCOVER is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with VDISCOVER. If not, see <http://www.gnu.org/licenses/>.

Copyright 2014 by G.Grieco
"""

import random
import gzip
import sys
import csv
import subprocess
import pickle
import numpy as np
import matplotlib as mpl

# hack from https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined to avoid using X
# mpl.use('Agg')
import matplotlib.pyplot as plt

from Utils import *
from Pipeline import *


# def Cluster(X, labels)
"""
  assert(len(X_red) == len(labels))

  from sklearn.cluster import MeanShift, estimate_bandwidth

  bandwidth = estimate_bandwidth(X, quantile=0.2)
  print "Clustering with bandwidth:", bandwidth

  af = MeanShift(bandwidth=bandwidth/1).fit(X_red)

  cluster_centers = af.cluster_centers_
  cluster_labels = af.labels_
  n_clusters = len(cluster_centers)

  plt.figure()

  for ([x,y],label, cluster_label) in zip(X_red,labels, cluster_labels):
    x = gauss(0,0.1) + x
    y = gauss(0,0.1) + y
    plt.scatter(x, y, c = colors[cluster_label % ncolors])
    #plt.text(x-0.05, y+0.01, label.split("/")[-1])

  for i,[x,y] in enumerate(cluster_centers):
    plt.plot(x, y, 'o', markerfacecolor=colors[i % ncolors],
             markeredgecolor='k', markersize=7)

  plt.title('Estimated number of clusters: %d' % n_clusters)
"""
# return zip(labels, cluster_labels)


batch_size = 25
window_size = 32
maxlen = window_size

embedding_dims = 5
nb_filters = 50
filter_length = 3
hidden_dims = 50
nb_epoch = 3


def ClusterCnn(model_file, train_file, valid_file, ftype, nsamples, outdir):

    f = open(model_file + ".pre")
    preprocessor = pickle.load(f)

    import h5py
    f = h5py.File(model_file + ".wei")

    layers = []
    for k in range(f.attrs['nb_layers']):
        g = f['layer_{}'.format(k)]
        layers.append([g['param_{}'.format(p)]
                       for p in range(g.attrs['nb_params'])])

    max_features = len(preprocessor.tokenizer.word_counts)

    print "Reading and sampling data to train.."
    train_programs, train_features, train_classes = read_traces(
        train_file, nsamples, cut=None)
    train_size = len(train_features)

    #y = train_programs
    X_train, y_train, labels = preprocessor.preprocess_traces(
        train_features, y_data=train_classes, labels=train_programs)
    new_model = make_cluster_cnn(
        "test",
        max_features,
        maxlen,
        embedding_dims,
        nb_filters,
        filter_length,
        hidden_dims,
        None,
        weights=layers)

    train_dict = dict()
    train_dict[ftype] = new_model.predict(X_train)

    model = make_cluster_pipeline_subtraces(ftype)
    X_red_comp = model.fit_transform(train_dict)
    explained_var = np.var(X_red_comp, axis=0)
    print explained_var

    X_red = X_red_comp[:, 0:2]
    X_red_next = X_red_comp[:, 2:4]

    colors = mpl.colors.cnames.keys()
    progs = list(set(labels))
    ncolors = len(colors)
    size = len(labels)
    print "Plotting.."

    for prog, [x, y] in zip(labels, X_red):
        # for prog,[x,y] in sample(zip(labels, X_red), min(size, 1000)):
        x = gauss(0, 0.05) + x
        y = gauss(0, 0.05) + y
        color = 'r'
        plt.scatter(x, y, c=color)

    """
  if valid_file is not None:
    valid_programs, valid_features, valid_classes = read_traces(valid_file, None, cut=None, maxsize=window_size) #None)
    valid_dict = dict()

    X_valid, _, valid_labels = preprocessor.preprocess_traces(valid_features, y_data=None, labels=valid_programs)
    valid_dict[ftype] = new_model.predict(X_valid)
    X_red_valid_comp = model.transform(valid_dict)

    X_red_valid = X_red_valid_comp[:,0:2]
    X_red_valid_next = X_red_valid_comp[:,2:4]

    for prog,[x,y] in zip(valid_labels, X_red_valid):
      x = gauss(0,0.05) + x
      y = gauss(0,0.05) + y
      plt.scatter(x, y, c='b')
      plt.text(x, y+0.02, prog.split("/")[-1])

  plt.show()
  """
    plt.savefig(train_file.replace(".gz", "") + ".png")
    print "Bandwidth estimation.."
    from sklearn.cluster import MeanShift, estimate_bandwidth

    X_red_sample = X_red[:min(size, 1000)]
    bandwidth = estimate_bandwidth(X_red_sample, quantile=0.2)
    print "Clustering with bandwidth:", bandwidth

    #X_red = np.vstack((X_red,X_red_valid))
    #X_red_next = np.vstack((X_red_next,X_red_valid_next))
    #labels = labels + valid_labels

    print X_red.shape, len(X_red), len(labels)
    # print valid_labels

    af = MeanShift(bandwidth=bandwidth / 1).fit(X_red)

    cluster_centers = af.cluster_centers_
    cluster_labels = af.labels_
    n_clusters = len(cluster_centers)

    plt.figure()
    for ([x, y], label, cluster_label) in zip(X_red, labels, cluster_labels):
        # for ([x,y],label, cluster_label) in sample(zip(X_red,labels,
        # cluster_labels), min(size, 1000)):
        x = gauss(0, 0.1) + x
        y = gauss(0, 0.1) + y
        plt.scatter(x, y, c=colors[cluster_label % ncolors])
        # print label
        # if label in valid_labels:
        #  plt.text(x-0.05, y+0.01, label.split("/")[-1])

    for i, [x, y] in enumerate(cluster_centers):
        plt.plot(x, y, 'o', markerfacecolor=colors[i % ncolors],
                 markeredgecolor='k', markersize=7)

    """
  #for prog,[x,y] in zip(valid_labels, X_red_valid):
    #x = gauss(0,0.1) + x
    #y = gauss(0,0.1) + y
    #plt.scatter(x, y, c='black')
    #plt.text(x, y+0.02, prog.split("/")[-1])


  plt.title('Estimated number of clusters: %d' % n_clusters)

  #plt.savefig("clusters.png")
  plt.show()
  """
    plt.savefig(train_file.replace(".gz", "") + ".clusters.png")

    clustered_traces = zip(labels, cluster_labels)
    writer = open_csv(train_file.replace(".gz", "") + ".clusters")
    for label, cluster in clustered_traces:
        writer.writerow([label, cluster])

    """

  clusters = dict()
  for label, cluster in clustered_traces:
    clusters[cluster] = clusters.get(cluster, []) + [label]

  for cluster, traces in clusters.items():
    plt.figure()
    plt.title('Cluster %d' % cluster)
    #X_clus = []

    #for prog in traces:
    #  i = labels.index(prog)
    #  X_clus.append(X_train[i])

    #train_dict = dict()
    #train_dict[ftype] = X_clus

    #model = make_cluster_pipeline_subtraces(ftype)
    #X_red = model.fit_transform(train_dict)

    #for [x,y],prog in zip(X_red,traces):
    for prog in traces:

      i = labels.index(prog)
      assert(i>=0)
      [x,y] = X_red_next[i]
      x = gauss(0,0.1) + x
      y = gauss(0,0.1) + y
      plt.scatter(x, y, c='r')

      #if prog in valid_labels:
      plt.text(x-0.05, y+0.01, prog.split("/")[-1])

      #plt.text(x, y+0.02, prog.split("/")[-1])

    plt.show()
    #plt.savefig('cluster-%d.png' % cluster)
  """

    # return clustered_traces


def TrainCnn(model_file, train_file, valid_file, ftype, nsamples):

    csvreader = open_csv(train_file)

    train_features = []
    train_programs = []
    train_classes = []

    train_programs, train_features, train_classes = read_traces(
        train_file, nsamples, cut=None)
    train_size = len(train_features)

    from keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(nb_words=None, filters="", lower=False, split=" ")
    # print type(train_features[0])
    tokenizer.fit_on_texts(train_features)
    max_features = len(tokenizer.word_counts)

    preprocessor = DeepReprPreprocessor(tokenizer, window_size, batch_size)
    X_train, y_train = preprocessor.preprocess(train_features, 10000)
    nb_classes = len(preprocessor.classes)
    print preprocessor.classes

    model = make_cluster_cnn(
        "train",
        max_features,
        maxlen,
        embedding_dims,
        nb_filters,
        filter_length,
        hidden_dims,
        nb_classes)
    model.fit(X_train, y_train, validation_split=0.1,
              batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)

    model.mypreprocessor = preprocessor
    #model_file = model_file + ".wei"
    #modelfile = open_model(model_file)
    print "Saving model to", model_file + ".wei"
    model.save_weights(model_file + ".wei")

    #model_file = model_file + ".pre"
    modelfile = open_model(model_file + ".pre")
    print "Saving preprocessor to", model_file + ".pre"
    # model.save_weights(model_file)
    modelfile.write(pickle.dumps(preprocessor, protocol=2))

"""
def ClusterDoc2Vec(model_file, train_file, valid_file, ftype, nsamples, param):

  train_programs, train_features, train_classes = read_traces(train_file, nsamples)
  train_size = len(train_programs)

  print "using", train_size,"examples to train."

  from gensim.models.doc2vec import TaggedDocument
  from gensim.models import Doc2Vec

  print "Vectorizing traces.."
  sentences = []

  for (prog,trace) in zip(train_programs,train_features):
     sentences.append(TaggedDocument(trace.split(" "), [prog]))

  model = Doc2Vec(dm=2, min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=8, iter=1)
  model.build_vocab(sentences)

  for epoch in range(20):
    #print model
    model.train(sentences)
    shuffle(sentences)

  train_dict = dict()

  vec_train_features = []
  for prog in train_programs:
    #print prog, model.docvecs[prog]
    vec_train_features.append(model.docvecs[prog])

  train_dict[ftype] = vec_train_features

  print "Transforming data and fitting model.."
  model = make_cluster_pipeline_doc2vec(ftype)
  X_red = model.fit_transform(train_dict)

  #mpl.rcParams.update({'font.size': 10})
  plt.figure()
  colors = 'brgcmykbgrcmykbgrcmykbgrcmyk'
  ncolors = len(colors)

  for prog,[x,y],cl in zip(train_programs, X_red, train_classes):
    x = gauss(0,0.1) + x
    y = gauss(0,0.1) + y
    try:
        plt.scatter(x, y, c=colors[int(cl)])
        plt.text(x, y+0.02, prog.split("/")[-1])
    except ValueError:
        plt.text(x, y+0.02, cl)

  #plt.show()
  plt.savefig(train_file.replace(".gz","")+".png")

  from sklearn.cluster import MeanShift, estimate_bandwidth

  bandwidth = estimate_bandwidth(X_red, quantile=0.2)
  print "Clustering with bandwidth:", bandwidth

  af = MeanShift(bandwidth=bandwidth*param).fit(X_red)

  cluster_centers = af.cluster_centers_
  labels = af.labels_
  n_clusters_ = len(cluster_centers)

  plt.close('all')
  plt.figure(1)
  plt.clf()

  for ([x,y],label, cluster_label) in zip(X_red,train_programs, labels):
    x = gauss(0,0.1) + x
    y = gauss(0,0.1) + y
    plt.scatter(x, y, c = colors[cluster_label % ncolors])

  for i,[x,y] in enumerate(cluster_centers):
    plt.plot(x, y, 'o', markerfacecolor=colors[i % ncolors],
             markeredgecolor='k', markersize=7)

  plt.title('Estimated number of clusters: %d' % n_clusters_)
  plt.savefig(train_file.replace(".gz","")+".clusters.png")

  #plt.show()

  clustered_traces = zip(train_programs, labels)
  writer = write_csv(train_file.replace(".gz","")+".clusters")
  for label, cluster in clustered_traces:
     writer.writerow([label.split("/")[-1], cluster])

"""


def ClusterScikit(
        model_file,
        train_file,
        valid_file,
        ftype,
        nsamples,
        vectorizer,
        reducer,
        param):

    train_programs, train_features, train_classes = read_traces(
        train_file, nsamples)
    train_size = len(train_programs)
    print "using", train_size, "examples to train."

    if vectorizer == "bow":

        train_dict = dict()
        train_dict[ftype] = train_features
        #batch_size = 16
        #window_size = 20

        print "Transforming data and fitting model.."
        model = make_cluster_pipeline_bow(ftype, reducer)
        X_red = model.fit_transform(train_dict)

    elif vectorizer == "doc2vec":

        from gensim.models.doc2vec import TaggedDocument
        from gensim.models import Doc2Vec

        print "Vectorizing traces.."
        sentences = []

        for (prog, trace) in zip(train_programs, train_features):
            sentences.append(TaggedDocument(trace.split(" "), [prog]))

        model = Doc2Vec(dm=2, min_count=1, window=5, size=100,
                        sample=1e-4, negative=5, workers=8, iter=1)
        model.build_vocab(sentences)

        for epoch in range(20):
            # print model
            model.train(sentences)
            shuffle(sentences)

        train_dict = dict()

        vec_train_features = []
        for prog in train_programs:
            # print prog, model.docvecs[prog]
            vec_train_features.append(model.docvecs[prog])

        train_dict[ftype] = vec_train_features

        print "Transforming data and fitting model.."
        model = make_cluster_pipeline_doc2vec(ftype, reducer)
        X_red = model.fit_transform(train_dict)

    #pl.rcParams.update({'font.size': 10})
    if isinstance(X_red, list):
        X_red = np.vstack(X_red)
        print X_red.shape

    if X_red.shape[1] == 2:

        plt.figure()
        colors = 'brgcmykbgrcmykbgrcmykbgrcmyk'
        ncolors = len(colors)

        for prog, [x, y], cl in zip(train_programs, X_red, train_classes):
            x = gauss(0, 0.1) + x
            y = gauss(0, 0.1) + y
            try:
                plt.scatter(x, y, c=colors[int(cl)])
                plt.text(x, y + 0.02, prog.split("/")[-1])
            except ValueError:
                plt.text(x, y + 0.02, cl)

        if valid_file is not None:
            valid_programs, valid_features, valid_classes = read_traces(
                valid_file, None)
            valid_dict = dict()
            valid_dict[ftype] = valid_features

            X_red = model.transform(valid_dict)
            for prog, [x, y], cl in zip(valid_programs, X_red, valid_classes):
                x = gauss(0, 0.1) + x
                y = gauss(0, 0.1) + y
                plt.scatter(x, y, c=colors[cl + 1])
                plt.text(x, y + 0.02, prog.split("/")[-1])

        # plt.show()
        plt.savefig(train_file.replace(".gz", "") + ".png")

    from sklearn.cluster import MeanShift, estimate_bandwidth

    bandwidth = estimate_bandwidth(X_red, quantile=0.2)
    print "Clustering with bandwidth:", bandwidth

    af = MeanShift(bandwidth=bandwidth * param).fit(X_red)

    cluster_centers = af.cluster_centers_
    labels = af.labels_
    n_clusters_ = len(cluster_centers)

    if X_red.shape[1] == 2:

        plt.close('all')
        plt.figure(1)
        plt.clf()

        for ([x, y], label, cluster_label) in zip(
                X_red, train_programs, labels):
            x = gauss(0, 0.1) + x
            y = gauss(0, 0.1) + y
            plt.scatter(x, y, c=colors[cluster_label % ncolors])

        for i, [x, y] in enumerate(cluster_centers):
            plt.plot(x, y, 'o', markerfacecolor=colors[i % ncolors],
                     markeredgecolor='k', markersize=7)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.savefig(train_file.replace(".gz", "") + ".clusters.png")

    # plt.show()

    clustered_traces = zip(train_programs, labels)
    writer = write_csv(train_file.replace(".gz", "") + ".clusters")
    for label, cluster in clustered_traces:
        writer.writerow([label.split("/")[-1], cluster])
