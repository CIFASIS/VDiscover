import random
import gzip
import sys
import csv
import subprocess
import pickle
import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from Pipeline import * 

csv.field_size_limit(sys.maxsize)
sys.setrecursionlimit(sys.maxsize)

def file_len(fname):

  if ".gz" in fname:
    cat = "zcat"
  else:
    cat = "cat"

  p = subprocess.Popen(cat + " " + fname + " | wc -l", shell=True, stdout=subprocess.PIPE, 
                                                                     stderr=subprocess.PIPE)
  result, err = p.communicate()
  if p.returncode != 0:
      raise IOError(err)
  return int(result.strip().split()[0])

def open_csv(in_file):
  
  if ".gz" in in_file:
    infile = gzip.open(in_file, "r")
  else:
    infile = open(in_file, "r")

  return csv.reader(infile, delimiter='\t')

def open_model(model_file):
  
  if ".pklz" in model_file:
    modelfile = gzip.open(model_file,"w+")
  else:
    modelfile = open(model_file,"w+")
 
  return modelfile


def PlotDeepRepr(model_file, train_file, valid_file, ftype, nsamples):

  f = gzip.open(model_file)
  old_model = pickle.load(f)
  preprocessor = old_model.mypreprocessor

  #print preprocessor.tokenizer
  #print preprocessor.tokenizer.word_counts
  max_features = len(preprocessor.tokenizer.word_counts)

  batch_size = 300
  window_size = 100
  maxlen = window_size

  embedding_dims = 20
  nb_filters = 250
  filter_length = 3
  hidden_dims = 250

  csvreader = open_csv(train_file) 

  train_features = []
  train_programs = []
  train_classes = [] 

  print "Reading and sampling data to train..",
  for i,col in enumerate(csvreader):

    program = col[0]
    features = col[1]
    if len(col) > 2:
      cl = col[2]
    else:
      cl = 0

    train_programs.append(program)
    train_features.append(features)
    train_classes.append(cl)

  train_size = len(train_features)

  #from keras.preprocessing.text import Tokenizer

  #tokenizer = Tokenizer(nb_words=None, filters="", lower=False, split=" ")
  #print type(train_features[0])
  #tokenizer.fit_on_texts(train_features)
  #max_features = len(tokenizer.word_counts)

  #preprocessor = FilterPreprocessor(tokenizer, window_size, batch_size)
  y = train_programs
  y = train_classes
  y = None
  X_train, labels = preprocessor.preprocess_traces(train_features, y, 300)
  #print X_train[5]

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
  new_model.add(Embedding(max_features, embedding_dims, weights=old_model.layers[0].get_weights()))
  new_model.add(Dropout(0.25))

  # we add a Convolution1D, which will learn nb_filters
  # word group filters of size filter_length:
  new_model.add(Convolution1D(input_dim=embedding_dims,
                        nb_filter=nb_filters,
                        filter_length=filter_length,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1,
                        weights=old_model.layers[2].get_weights()))

  # we use standard max pooling (halving the output of the previous layer):
  new_model.add(MaxPooling1D(pool_length=2))

  # We flatten the output of the conv layer, so that we can add a vanilla dense layer:
  new_model.add(Flatten())

  # Computing the output shape of a conv layer can be tricky;
  # for a good tutorial, see: http://cs231n.github.io/convolutional-networks/
  output_size = nb_filters * (((maxlen - filter_length) / 1) + 1) / 2

  # We add a vanilla hidden layer:
  new_model.add(Dense(output_size, hidden_dims, weights=old_model.layers[5].get_weights()))
  #new_model.add(Dropout(0.25))
  #new_model.add(Activation('relu'))

  new_model.compile(loss='mean_squared_error', optimizer='rmsprop')

  train_dict = dict()
  train_dict[ftype] = new_model._predict(X_train)

  model = make_cluster_pipeline_subtraces(ftype)
  X_red = model.fit_transform(train_dict)

  plt.figure()
  print len(X_red), len(labels)

  for ([x,y],label) in zip(X_red,labels):
    x = gauss(0,0.05) + x
    y = gauss(0,0.05) + y
    plt.scatter(x, y)
    plt.text(x, y+0.2, label)


  plt.show()
 

def TrainDeepRepr(model_file, train_file, valid_file, ftype, nsamples):
 
  csvreader = open_csv(train_file) 

  train_features = []
  train_programs = []
  train_classes = []

  batch_size = 100
  window_size = 100
  maxlen = window_size

  embedding_dims = 20
  nb_filters = 250
  filter_length = 3
  hidden_dims = 250
  nb_epoch = 50
 
  print "Reading and sampling data to train..",
  for i,col in enumerate(csvreader):

    program = col[0]
    features = col[1]
    cl = 0
    #pfeatures = features.split(" ")[:-1]
    #r = random.randint(1, len(pfeatures)-1)

    train_programs.append(program)
    train_features.append(features)
    #cl = pfeatures[r].split(":")[0]
    train_classes.append(cl)

  #assert(0)
   
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

  sys.setrecursionlimit(sys.maxsize)
  model_file = "cluster.pklz"
  modelfile = open_model(model_file)
  print "Saving model to",model_file
  modelfile.write(pickle.dumps(model, protocol=2))

def ClusterScikit(model_file, train_file, valid_file, ftype, nsamples):
 
  import matplotlib.pyplot as plt
  import matplotlib as mpl

  csvreader = open_csv(train_file) 
  train_features = []
  train_programs = []
  train_classes = []
 
  print "Reading and sampling data to train..",
  if nsamples is None:
    for i,col in enumerate(csvreader):

      program = col[0]
      features = col[1] 
      if len(col) > 2:
        cl = int(col[2])
      else:
        cl = 0

      train_programs.append(program)
      train_features.append(features)
      train_classes.append(cl)
  else:
    
    train_size = file_len(train_file)
    skip_until = randint(0,train_size - nsamples)

    for i,col in enumerate(csvreader):
 
      if i < skip_until:
        continue
      elif i - skip_until == nsamples:
        break

      program = col[0]
      features = col[1] 

      train_programs.append(program)
      train_features.append(features)

      if len(col) > 2:
        cl = int(col[2])
      else:
        cl = 0

      train_classes.append(cl)
  train_size = len(train_features)

  assert(train_size == len(train_classes))

  print "using", train_size,"examples to train."

  train_dict = dict()
  train_dict[ftype] = train_features
  batch_size = 16
  window_size = 20

  from sklearn.cluster import MeanShift

  print "Transforming data and fitting model.."
  model = make_cluster_pipeline_bow(ftype)
  train_X = model.fit_transform(train_dict)

  mpl.rcParams.update({'font.size': 10})
  plt.figure()
  colors = 'brgcmykbgrcmykbgrcmykbgrcmyk'


  #i = 0
  for prog,[x,y],cl in zip(train_programs, train_X, train_classes):
    #x = gauss(0,0.2) + x
    #y = gauss(0,0.2) + y
    plt.scatter(x, y, c=colors[cl])
    plt.text(x, y+0.2, prog.split("-")[-1])
    #i = i + 1

  
  if valid_file is not None:
    csvreader = open_csv(valid_file) 

    valid_features = []
    valid_programs = []
    valid_classes = []
  
    print "Reading data to valid..",
    for i,(program, features) in enumerate(csvreader):
      valid_programs.append(program)
      valid_features.append(features)
      valid_classes.append(int(cl))

    print "using", len(train_features),"examples to valid."
    #X_valid,y_valid = preprocessor.preprocess(valid_features, valid_classes)
  else:
    plt.show()
    return

  valid_dict = dict()
  valid_dict[ftype] = valid_features
  valid_X = model.fit_transform(valid_dict)

  for [x,y] in valid_X:
    rx = gauss(0,0.2) + x
    ry = gauss(0,0.2) + y
    
    plt.scatter(rx,ry, c='r')
    #plt.text(x, y, valid_programs[i])

  plt.show()
  return

  af = MeanShift().fit(X) 

  cluster_centers = af.cluster_centers_
  labels = af.labels_
  n_clusters_ = len(cluster_centers)

  plt.close('all')
  plt.figure(1)
  plt.clf()

  for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    
  
  plt.title('Estimated number of clusters: %d' % n_clusters_)
  plt.show()


def Cluster(model_file, out_file, train_file, valid_file, ttype, ftype, nsamples):

  if ttype == "cluster":

    ClusterScikit(out_file, train_file, valid_file, ftype, nsamples)
    return

    try:
      import keras
    except:
      print "Failed to import keras modules to perform LSTM training"
      return

    if model_file is None:      
      TrainDeepRepr(out_file, train_file, valid_file, ftype, nsamples)
    else:
      PlotDeepRepr(model_file, train_file, valid_file, ftype, nsamples)
