import random
import gzip
import sys
import csv
import subprocess
import pickle

from Pipeline import * 
from sklearn.metrics import confusion_matrix

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


def TrainScikitLearn(model_file, train_file, valid_file, ftype, nsamples):

  csvreader = open_csv(train_file) 
  modelfile = open_model(model_file)

  train_features = []
  train_programs = []
  train_classes = []
  
  print "Reading and sampling data to train..",
  if nsamples is None:
    for i,(program, features, cl) in enumerate(csvreader):
      train_programs.append(program)
      train_features.append(features)
      train_classes.append(int(cl))
  else:
    
    train_size = file_len(train_file)
    skip_until = randint(0,train_size - nsamples)

    for i,(program, features, cl) in enumerate(csvreader):
 
      if i < skip_until:
        continue
      elif i - skip_until == nsamples:
        break

      train_programs.append(program)
      train_features.append(features)
      train_classes.append(int(cl))
  
  print "using", len(train_features),"examples to train."

  train_dict = dict()
  train_dict[ftype] = train_features 

  print "Transforming data and fitting model.."
  model = make_train_pipeline(ftype)
  model.fit(train_dict,train_classes)

  print "Resulting model:"
  print model
  print confusion_matrix(train_classes, model.predict(train_dict))

  print "Saving model to",model_file
  modelfile.write(pickle.dumps(model))

  #outfile = open(out_file, "a+")
  #csvwriter = csv.writer(outfile, delimiter='\t')

  #model = pickle.load(gzip.open(model_file))

def TrainKeras(model_file, train_file, valid_file, ftype, nsamples):
 
  csvreader = open_csv(train_file) 
  modelfile = open_model(model_file)

  train_features = []
  train_programs = []
  train_classes = []
  
  print "Reading and sampling data to train..",
  if nsamples is None:
    for i,(program, features, cl) in enumerate(csvreader):
      train_programs.append(program)
      train_features.append(features)
      train_classes.append(int(cl))
  else:
    
    train_size = file_len(in_file)
    skip_until = random.randint(0,train_size - nsamples)

    for i,(program, features, cl) in enumerate(csvreader):
 
      if i < skip_until:
        continue
      elif i - skip_until == nsamples:
        break

      train_programs.append(program)
      train_features.append(features)
      train_classes.append(int(cl))
  train_size = len(train_features)

  assert(train_size == len(train_classes))

  print "using", train_size,"examples to train."

  train_dict = dict()
  train_dict[ftype] = train_features
  batch_size = 16
  window_size = 25 

  from keras.preprocessing.text import Tokenizer

  tokenizer = Tokenizer(nb_words=None, filters="", lower=False, split=" ")
  #print type(train_features[0])
  tokenizer.fit_on_texts(train_features)
  max_features = len(tokenizer.word_counts)

  preprocessor = KerasPreprocessor(tokenizer, window_size, batch_size)

  if valid_file is not None:
    csvreader = open_csv(valid_file) 

    valid_features = []
    valid_programs = []
    valid_classes = []
  
    print "Reading data to valid..",
    for i,(program, features, cl) in enumerate(csvreader):
      valid_programs.append(program)
      valid_features.append(features)
      valid_classes.append(int(cl))

    print "using", len(train_features),"examples to valid."
    #X_valid,y_valid = preprocessor.preprocess(valid_features, valid_classes)
  else:
    valid_features,train_features = train_features[0:int(0.1*train_size)], train_features[int(0.1*train_size):]
    valid_classes,train_classes = train_classes[0:int(0.1*train_size)], train_classes[int(0.1*train_size):]

  X_valid,y_valid = preprocessor.preprocess(valid_features, valid_classes, 500)
  X_train,y_train = preprocessor.preprocess(train_features, train_classes, 10000)

  from keras.models import Sequential
  from keras.layers.core import Dense, Dropout, Activation
  from keras.layers.embeddings import Embedding
  from keras.layers.recurrent import LSTM, GRU
  from keras.optimizers import Adam

  print "Creating and compiling a LSTM.."
  model = Sequential()
  model.add(Embedding(max_features, 10))
  model.add(LSTM(10, 32)) 
  model.add(Dropout(0.50))
  model.add(Dense(32, 1))
  model.add(Activation('sigmoid'))

  # try using different optimizers and different optimizer config
  opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, kappa=1-1e-8)
  model.compile(loss='binary_crossentropy', optimizer=opt, class_mode="binary")
  #model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=30, validation_data = (X_valid,y_valid), show_accuracy=True)
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5, show_accuracy=True)

  print "Saving model to",model_file
  
  modelfile.write(pickle.dumps(KerasPredictor(preprocessor,model,ftype)))


def Train(model_file, train_file, valid_file, ttype, ftype, nsamples):
  if ttype == "rf":
    TrainScikitLearn(model_file, train_file, valid_file, ftype, nsamples)

  elif ttype == "lstm":
     try:
       import keras
     except:
       print "Failed to import keras modules to perform LSTM training"
       return
     TrainKeras(model_file, train_file, valid_file, ftype, nsamples)

