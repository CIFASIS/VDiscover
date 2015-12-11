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
import matplotlib.pyplot as plt
import matplotlib as mpl
#import pylab as plb

from Utils import *
from Pipeline import *

"""
def Cluster(X, labels)

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
  
  return zip(labels, cluster_labels)
"""


def ClusterConv(model_file, train_file, valid_file, ftype, nsamples, outdir):

  f = open(model_file+".pre")
  preprocessor = pickle.load(f)

  import h5py
  f = h5py.File(model_file+".wei")

  layers = []
  for k in range(f.attrs['nb_layers']):
            g = f['layer_{}'.format(k)]
            layers.append([g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])])

  max_features = len(preprocessor.tokenizer.word_counts)

  batch_size = 100
  window_size = 300
  maxlen = window_size

  embedding_dims = 20
  nb_filters = 50
  filter_length = 3
  hidden_dims = 250

  #csvreader = load_csv(train_file)
  print "Reading and sampling data to train.."
  train_programs, train_features, train_classes = read_traces(train_file, nsamples, cut=1, maxsize=window_size)
  train_size = len(train_features)

  #y = train_programs
  X_train, y_train, labels = preprocessor.preprocess_traces(train_features, y_data=train_classes, labels=train_programs)

  from keras.preprocessing import sequence
  from keras.optimizers import RMSprop
  from keras.models import Sequential
  from keras.layers.core import Dense, Dropout, Activation, Flatten
  from keras.layers.embeddings import Embedding
  from keras.layers.convolutional import Convolution1D, MaxPooling1D

  print('Build model...')
  new_model = Sequential()

  # we start off with an efficient embedding layer which maps
  # our vocab indices into embedding_dims dimensions
  new_model.add(Embedding(max_features, embedding_dims, weights=layers[0]))
  new_model.add(Dropout(0.25))

  # we add a Convolution1D, which will learn nb_filters
  # word group filters of size filter_length:
  new_model.add(Convolution1D(input_dim=embedding_dims,
                        nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1,
                        weights=layers[2]))

  # we use standard max pooling (halving the output of the previous layer):
  new_model.add(MaxPooling1D(pool_length=2))

  # We flatten the output of the conv layer, so that we can add a vanilla dense layer:
  new_model.add(Flatten())

  # Computing the output shape of a conv layer can be tricky;
  # for a good tutorial, see: http://cs231n.github.io/convolutional-networks/
  output_size = nb_filters * (((maxlen - filter_length) / 1) + 1) / 2

  # We add a vanilla hidden layer:
  new_model.add(Dense(output_size, hidden_dims, weights=layers[5]))
  #new_model.add(Dropout(0.25))
  #new_model.add(Activation('relu'))

  new_model.compile(loss='mean_squared_error', optimizer='rmsprop')

  train_dict = dict()
  train_dict[ftype] = new_model._predict(X_train)

  model = make_cluster_pipeline_subtraces(ftype)
  X_red_comp = model.fit_transform(train_dict)
  explained_var = np.var(X_red_comp, axis=0)
  print explained_var

  X_red = X_red_comp[:,0:2]
  X_red_next = X_red_comp[:,2:4]

  colors = mpl.colors.cnames.keys() #'rbgcmykbgrcmykbgrcmykbgrcmyk'
  progs = list(set(labels))
  ncolors = len(colors)

  for prog,[x,y] in zip(labels, X_red):
    #x = gauss(0,0.1) + x
    #y = gauss(0,0.1) + y
    color = 'r' #colors[progs.index(prog)]
    plt.scatter(x, y, c=color )
    #plt.text(x, y+0.02, prog.split("/")[-1])


  if valid_file is not None: 
    valid_programs, valid_features, valid_classes = read_traces(valid_file, None, cut=1, maxsize=window_size) #None)
    valid_dict = dict()

    X_valid, _, valid_labels = preprocessor.preprocess_traces(valid_features, y_data=None, labels=valid_programs)
    valid_dict[ftype] = new_model._predict(X_valid) 
    X_red_valid_comp = model.transform(valid_dict)

    X_red_valid = X_red_valid_comp[:,0:2]
    X_red_valid_next = X_red_valid_comp[:,2:4]

    for prog,[x,y] in zip(valid_labels, X_red_valid):
      x = gauss(0,0.1) + x
      y = gauss(0,0.1) + y
      plt.scatter(x, y, c='b')
      plt.text(x, y+0.02, prog.split("/")[-1])

  plt.show()
  #plt.savefig("plot.png")
  #return None

  from sklearn.cluster import MeanShift, estimate_bandwidth

  bandwidth = estimate_bandwidth(X_red, quantile=0.2)
  print "Clustering with bandwidth:", bandwidth

  #X_red = np.vstack((X_red,X_red_valid))
  #X_red_next = np.vstack((X_red_next,X_red_valid_next))
  #labels = labels + valid_labels

  print X_red.shape, len(X_red), len(labels)
  #print valid_labels

  af = MeanShift(bandwidth=bandwidth/5).fit(X_red)

  cluster_centers = af.cluster_centers_
  cluster_labels = af.labels_
  n_clusters = len(cluster_centers)

  plt.figure()
  for ([x,y],label, cluster_label) in zip(X_red,labels, cluster_labels):
    x = gauss(0,0.1) + x
    y = gauss(0,0.1) + y
    plt.scatter(x, y, c = colors[cluster_label % ncolors])
    #print label
    #if label in valid_labels:
    #  plt.text(x-0.05, y+0.01, label.split("/")[-1])

  for i,[x,y] in enumerate(cluster_centers):
    plt.plot(x, y, 'o', markerfacecolor=colors[i % ncolors],
             markeredgecolor='k', markersize=7)

  #for prog,[x,y] in zip(valid_labels, X_red_valid):
    #x = gauss(0,0.1) + x
    #y = gauss(0,0.1) + y
    #plt.scatter(x, y, c='black')
    #plt.text(x, y+0.02, prog.split("/")[-1])


  plt.title('Estimated number of clusters: %d' % n_clusters)

  #plt.savefig("clusters.png")
  plt.show()
  clustered_traces = zip(labels, cluster_labels)

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


  
  return clustered_traces
  #csvwriter = open_csv(train_file+".clusters")
  #for (label, cluster_label) in zip(labels, cluster_labels):
  #  csvwriter.writerow([label, cluster_label])

  #print "Clusters dumped!"


def TrainDeepRepr(model_file, train_file, valid_file, ftype, nsamples):

  csvreader = open_csv(train_file)

  train_features = []
  train_programs = []
  train_classes = []

  batch_size = 100
  window_size = 300
  maxlen = window_size

  embedding_dims = 20
  nb_filters = 250
  filter_length = 3
  hidden_dims = 250
  nb_epoch = 1

  train_programs, train_features, train_classes = read_traces(train_file, nsamples, cut=None)
  train_size = len(train_features)

  from keras.preprocessing.text import Tokenizer

  tokenizer = Tokenizer(nb_words=None, filters="", lower=False, split=" ")
  #print type(train_features[0])
  tokenizer.fit_on_texts(train_features)
  max_features = len(tokenizer.word_counts)

  preprocessor = DeepReprPreprocessor(tokenizer, window_size, batch_size)
  X_train,y_train = preprocessor.preprocess(train_features, 3000)
  nb_classes = len(preprocessor.classes)
  print preprocessor.classes
  #print X_train[0], len(X_train[0])
  #print X_train[1], len(X_train[1])

  #print set(y_train)
  #assert(0)

  from keras.preprocessing import sequence
  from keras.optimizers import RMSprop
  from keras.models import Sequential
  from keras.layers.core import Dense, Dropout, Activation, Flatten
  from keras.layers.embeddings import Embedding
  from keras.layers.convolutional import Convolution1D, MaxPooling1D

  print('Build model...')
  model = Sequential()

  # we start off with an efficient embedding layer which maps
  # our vocab indices into embedding_dims dimensions
  model.add(Embedding(max_features, embedding_dims))
  model.add(Dropout(0.25))

  # we add a Convolution1D, which will learn nb_filters
  # word group filters of size filter_length:
  model.add(Convolution1D(input_dim=embedding_dims,
                        nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))

  # we use standard max pooling (halving the output of the previous layer):
  model.add(MaxPooling1D(pool_length=2))

  # We flatten the output of the conv layer, so that we can add a vanilla dense layer:
  model.add(Flatten())

  # Computing the output shape of a conv layer can be tricky;
  # for a good tutorial, see: http://cs231n.github.io/convolutional-networks/
  output_size = nb_filters * (((maxlen - filter_length) / 1) + 1) / 2

  # We add a vanilla hidden layer:
  model.add(Dense(output_size, hidden_dims))
  model.add(Dropout(0.25))
  model.add(Activation('relu'))

  # We project onto a single unit output layer, and squash it with a sigmoid:
  model.add(Dense(hidden_dims, nb_classes))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode="categorical")
  model.fit(X_train, y_train, validation_split=0.1, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)

  model.mypreprocessor = preprocessor
  model_file = "cluster-weights.hdf5"
  #modelfile = open_model(model_file)
  print "Saving model to",model_file
  model.save_weights(model_file)

  model_file = "cluster-preprocessor.pklz"
  modelfile = open_model(model_file)
  print "Saving preprocessor to",model_file
  #model.save_weights(model_file)
  modelfile.write(pickle.dumps(preprocessor, protocol=2))


def ClusterScikit(model_file, train_file, valid_file, ftype, nsamples):

  #import matplotlib.pyplot as plt
  #import matplotlib as mpl

  #csvreader = open_csv(train_file)
  train_programs, train_features, train_classes = read_traces(train_file, nsamples)
  train_size = len(train_programs)

  print "using", train_size,"examples to train."

  train_dict = dict()
  train_dict[ftype] = train_features
  #batch_size = 16
  #window_size = 20

  #from sklearn.cluster import MeanShift

  print "Transforming data and fitting model.."
  model = make_cluster_pipeline_bow(ftype)
  X_red = model.fit_transform(train_dict)

  #mpl.rcParams.update({'font.size': 10})
  plt.figure()
  colors = 'brgcmykbgrcmykbgrcmykbgrcmyk'

  for prog,[x,y],cl in zip(train_programs, X_red, train_classes):
    x = gauss(0,0.1) + x
    y = gauss(0,0.1) + y
    plt.scatter(x, y, c=colors[cl])
    plt.text(x, y+0.02, prog.split("/")[-1])


  if valid_file is not None: 
    valid_programs, valid_features, valid_classes = read_traces(valid_file, None)
    valid_dict = dict()
    valid_dict[ftype] = valid_features
 
    X_red = model.transform(valid_dict)
    for prog,[x,y],cl in zip(valid_programs, X_red, valid_classes):
      x = gauss(0,0.1) + x
      y = gauss(0,0.1) + y
      plt.scatter(x, y, c=colors[cl+1])
      plt.text(x, y+0.02, prog.split("/")[-1])

  plt.show()
  #af = MeanShift().fit(X_red)

  #cluster_centers = af.cluster_centers_
  #labels = af.labels_
  #n_clusters_ = len(cluster_centers)

  #plt.close('all')
  #plt.figure(1)
  #plt.clf()

  #for k, col in zip(range(n_clusters_), colors):
  #  my_members = labels == k
  #  cluster_center = cluster_centers[k]
  #  plt.plot(X_red[my_members, 0], X_red[my_members, 1], col + '.')
  #  plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
  #           markeredgecolor='k', markersize=14)


  #plt.title('Estimated number of clusters: %d' % n_clusters_)
  #plt.show()

def Cluster(train_file, valid_file, ftype, nsamples):

  ClusterScikit(None, train_file, valid_file, ftype, nsamples)

  #if ttype == "cluster":
    #ClusterScikit(out_file, train_file, valid_file, ftype, nsamples)

    #try:
    #  import keras
    #except:
    #  print "Failed to import keras modules to perform LSTM training"
    #  return

    #if model_file is None:
    #  TrainDeepRepr(out_file, train_file, valid_file, ftype, nsamples)
    #else:
    #  PlotDeepRepr(model_file, train_file, valid_file, ftype, nsamples, outfile)
