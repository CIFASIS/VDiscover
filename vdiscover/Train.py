import random
import gzip
import sys
import csv
import subprocess
import pickle

from Pipeline import * 

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

csv.field_size_limit(sys.maxsize)

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


def make_pipeline(ftype):
  
  if ftype is "dynamic":
    return Pipeline(steps=[
         ('selector', ItemSelector(key='dynamic')),
         ('dvectorizer', CountVectorizer(tokenizer=static_tokenizer, ngram_range=(2,2), lowercase=False)),
         ('todense', DenseTransformer()),
         ('classifier', RandomForestClassifier(n_estimators=1000, max_features=None, max_depth=100))
    ])
  elif ftype is "static":
    return Pipeline(steps=[
         ('selector', ItemSelector(key='static')),
         ('dvectorizer', CountVectorizer(tokenizer=static_tokenizer, ngram_range=(1,1), lowercase=False)),
         ('todense', DenseTransformer()),
         ('classifier', LogisticRegression(penalty="l2", C=1e-07, tol=1e-06))
    ])
  else:
    assert(0)

def Train(model_file, in_file, ftype, nsamples):
 
  if ".gz" in in_file:
    infile = gzip.open(in_file, "r")
  else:
    infile = open(in_file, "r")
  
  if ".pklz" in model_file:
    modelfile = gzip.open(model_file,"w+")
  else:
    modelfile = open(model_file,"w+")
 
  csvreader = csv.reader(infile, delimiter='\t')

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
  
  print "using", len(train_features),"examples to train."

  train_dict = dict()
  train_dict[ftype] = train_features 

  print "Transforming data and fitting model.."
  model = make_pipeline(ftype)
  model.fit(train_dict,train_classes)

  print "Resulting model:"
  print model 
  print "Saving model to",model_file
  modelfile.write(pickle.dumps(model))

  #outfile = open(out_file, "a+")
  #csvwriter = csv.writer(outfile, delimiter='\t')

  #model = pickle.load(gzip.open(model_file))

