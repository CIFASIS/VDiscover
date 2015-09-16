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

import gzip
import subprocess
import pickle
import csv
import random

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

def load_model(model_file):
  
  if ".pklz" in model_file:
    modelfile = gzip.open(model_file,"r")
  else:
    modelfile = open(model_file,"r")
  
  model = pickle.load(gzip.open(model_file))
  return model

def open_model(model_file):
  
  if ".pklz" in model_file:
    modelfile = gzip.open(model_file,"w+")
  else:
    modelfile = open(model_file,"w+")
 
  return modelfile

def read_traces(csvreader, train_file, nsamples, cut=None, maxsize=50):

  train_features = []
  train_programs = []
  train_classes = []
  
  #print "Reading and sampling data to train..",
  if nsamples is None:
    for i,col in enumerate(csvreader):

      program = col[0]
      features = col[1]
      if len(col) > 2:
        cl = int(col[2])
      else:
        cl = -1

      raw_trace = features[:-1]
      trace = raw_trace.split(" ")
      size = len(trace)

      if cut is None or size < maxsize:

        train_programs.append(program)
        train_features.append(features)
        train_classes.append(cl)
      else:
        for _ in range(cut):

          #start = random.randint(0,size/2)
          #end = random.randint(size/2+1, size)
          start = random.randint(0,size)
          end = start + maxsize
 
          features = " ".join(trace[start:end+1]) 
 
          train_programs.append(program)
          train_features.append(features)
          train_classes.append(cl)
  else:
    
    train_size = file_len(train_file)
    skip_until = random.randint(0,train_size - nsamples)

    for i,col in enumerate(csvreader):
 
      if i < skip_until:
        continue
      elif i - skip_until == nsamples:
        break

      program = col[0]
      features = col[1]
      if len(col) > 2:
        cl = int(col[2])
      else:
        cl = -1

      raw_trace = features[:-1]
      trace = raw_trace.split(" ")
      size = len(trace)

      if cut is None or size < maxsize:

        train_programs.append(program)
        train_features.append(features)
        train_classes.append(cl)
      else:
        for _ in range(cut):

          #start = random.randint(0,size/2)
          #end = random.randint(size/2+1, size)
          start = random.randint(0,size-2)
          end = start + random.randint(1,size-1)
 
          features = " ".join(trace[start:end+1]) 
 
          train_programs.append(program)
          train_features.append(features)
          train_classes.append(cl)


  return train_programs, train_features, train_classes 
