#! /usr/bin/python2.7

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

import sys
import os
import csv
import gzip
import random
import cPickle
import gc

import numpy
import numpy.random

from utils import *

from sklearn import cross_validation

url = "https://github.com/CIFASIS/VDiscover/releases/download/0.1/vdiscovery.tar"
cache_dir = "cache"

data_dir = sys.argv[1]
seed = int(sys.argv[2])
data_dir = os.path.join(data_dir,str(hash(seed)))

data_filename = os.path.join(cache_dir,"static_features.csv.gz")
vuln_filename = os.path.join(cache_dir,"vulnerable_programs.csv.gz")

random.seed(seed)
csv.field_size_limit(sys.maxsize)

norm_programs = dict()
vuln_programs = dict()
train_size = 5000
valid_size = 10

cache_filename = os.path.join(data_dir,"cache.pkl")

if not os.path.isfile(os.path.join(cache_dir,"vdiscovery.tar")):
  os.system("mkdir -p "+cache_dir)
  os.system("wget '" + url + "' -O "+ cache_dir+"/vdiscovery.tar")
  os.system("cd "+cache_dir+" ; tar -xf vdiscovery.tar")

if not os.path.isfile(cache_filename):
  os.system("mkdir -p "+data_dir)
  # first, we read the list of vulnerable programs
  with gzip.open(vuln_filename, 'rb') as csvfile1:
    reader1 = csv.reader(csvfile1, delimiter='\t')

    for row in reader1:
      vuln_programs[row[0]] = [] 

  limit = sys.maxint

  # we read the list of all programs
  with gzip.open(data_filename, 'rb') as csvfile1:
    reader1 = csv.reader(csvfile1, delimiter='\t')
    
    for i,row in enumerate(reader1):
      if (len(row) <> 3) or (not ("/" in row[0])):
        continue
           
      if  (row[0] in vuln_programs):
        vuln_programs[row[0]].append(i)
      elif  (row[0] in norm_programs):
        norm_programs[row[0]].append(i)
      else:
        norm_programs[row[0]] = [i] 

  cPickle.dump([vuln_programs,norm_programs],open(cache_filename,"w+")) 

else:
  vuln_programs, norm_programs = cPickle.load(open(cache_filename))

vuln_size = len(vuln_programs)
norm_size = len(norm_programs)

print vuln_size, norm_size

vuln_traces_per_program = int (train_size / float(vuln_size))
norm_traces_per_program = int (train_size / float(norm_size))

valid_vuln_traces_per_program = int ((valid_size / float(vuln_size)) * 2.5)  
valid_norm_traces_per_program = int ((valid_size / float(norm_size)) * 2.5)

train_prop = 0.99
n_folds = 2

vuln_sample = random.sample(vuln_programs.keys(), vuln_size)
norm_sample = random.sample(norm_programs.keys(), norm_size)

train_vuln_programs = vuln_sample[:int(train_prop*vuln_size)]
#valid_vuln_programs  = vuln_sample[int(train_prop*vuln_size):(int(train_prop*vuln_size)+int(valid_prop*vuln_size))] 
test_vuln_programs  = vuln_sample[int(train_prop*vuln_size):]

train_norm_programs = norm_sample[:int(train_prop*norm_size)]
#valid_norm_programs  = norm_sample[int(train_prop*norm_size):(int(train_prop*norm_size)+int(valid_prop*norm_size))] 
test_norm_programs  = norm_sample[int(train_prop*norm_size):] 


def dump(train_vuln_programs, train_norm_programs, test_vuln_programs, test_norm_programs):

  norm_kf = cross_validation.KFold(len(train_norm_programs),  n_folds=n_folds)
  vuln_kf = cross_validation.KFold(len(train_vuln_programs),  n_folds=n_folds)

  for k, (norm_f, vuln_f) in enumerate(zip(norm_kf,vuln_kf)):
    print k
    
    cdata_dir = os.path.join(data_dir,str(k))
    os.system("mkdir -p "+   cdata_dir)

    ctrain_vuln_programs = [train_vuln_programs[i] for i in vuln_f[0]]
    cvalid_vuln_programs = [train_vuln_programs[i] for i in vuln_f[1]]
    ctest_vuln_programs = test_vuln_programs
  
    #print ctrain_vuln_programs
    #print cvalid_vuln_programs

    train_vuln_traces = dict()
    valid_vuln_traces = dict()
    test_vuln_traces = dict()

    for prog in ctrain_vuln_programs:
      size = len(vuln_programs[prog])
      sample = random.sample(vuln_programs[prog], size)

      assert(size > 0)

      for _ in range(vuln_traces_per_program):
        trace = sample[numpy.random.randint(size)]
        if trace in train_vuln_traces:
          train_vuln_traces[trace] = train_vuln_traces[trace] + 1
        else: 
          train_vuln_traces[trace] = 1

    for prog in cvalid_vuln_programs:

      size = len(vuln_programs[prog])
      sample = random.sample(vuln_programs[prog], size)

      assert(size > 0)

      for _ in range(valid_vuln_traces_per_program):
        trace = sample[numpy.random.randint(size)]
        if trace in valid_vuln_traces:
          valid_vuln_traces[trace] = valid_vuln_traces[trace] + 1
        else: 
          valid_vuln_traces[trace] = 1

      #for trace in vuln_programs[prog]: 
        #ts = dict(zip(traces, [1]*len(traces))) #set(vuln_programs[prog])
        #valid_vuln_traces[trace] = 1

    for prog in ctest_vuln_programs:
      for trace in vuln_programs[prog]: 
        #ts = dict(zip(traces, [1]*len(traces))) #set(vuln_programs[prog])
        test_vuln_traces[trace] = 1

    ctrain_norm_programs = [train_norm_programs[i] for i in norm_f[0]]
    cvalid_norm_programs = [train_norm_programs[i] for i in norm_f[1]]
    ctest_norm_programs = test_norm_programs

    train_norm_traces = dict()
    valid_norm_traces = dict()
    test_norm_traces  = dict()

    for prog in ctrain_norm_programs:
      size = len(norm_programs[prog])
      sample = random.sample(norm_programs[prog], len(norm_programs[prog]))

      assert(size > 0)

      for _ in range(norm_traces_per_program):
        trace = sample[numpy.random.randint(size)]
        if trace in train_norm_traces:
          train_norm_traces[trace] = train_norm_traces[trace] + 1
        else: 
          train_norm_traces[trace] = 1

    for prog in cvalid_norm_programs:

      size = len(norm_programs[prog])
      sample = random.sample(norm_programs[prog], len(norm_programs[prog]))

      assert(size > 0)

      for _ in range(valid_norm_traces_per_program):
        trace = sample[numpy.random.randint(size)]
        if trace in valid_norm_traces:
          valid_norm_traces[trace] = valid_norm_traces[trace] + 1
        else: 
          valid_norm_traces[trace] = 1

      #for trace in norm_programs[prog]:
        #ts = dict(zip(traces, [1]*len(traces))) 
        #valid_norm_traces[trace] = 1

    for prog in ctest_norm_programs:
      for trace in norm_programs[prog]:
        #ts = dict(zip(traces, [1]*len(traces))) 
        test_norm_traces[trace] = 1

    train_filename =  os.path.join(cdata_dir,"train.csv.gz")
    tmp_filename =  os.path.join(cdata_dir,"tmp.csv.gz")
    buggy_valid_filename =  os.path.join(cdata_dir,"buggy_valid.csv.gz")
    robust_valid_filename =  os.path.join(cdata_dir,"robust_valid.csv.gz")
    buggy_test_filename =  os.path.join(cdata_dir,"buggy_test.csv.gz")
    robust_test_filename =  os.path.join(cdata_dir,"robust_test.csv.gz")

    #seed_filename = os.path.join(cdata_dir,"seed")

    # we read the list of all programs and traces
    with gzip.open(data_filename, 'rb') as csvfile1:
      with gzip.open(tmp_filename, 'wb') as csvfile2:
        with gzip.open(buggy_valid_filename, 'wb') as csvfile3:
          with gzip.open(robust_valid_filename, 'wb') as csvfile4:
            with gzip.open(buggy_test_filename, 'wb') as csvfile5:
              with gzip.open(robust_test_filename, 'wb') as csvfile6:



                traces_csv = csv.reader(csvfile1, delimiter='\t')
                train_csv = csv.writer(csvfile2, delimiter='\t')

                buggy_valid_csv = csv.writer(csvfile3, delimiter='\t')
                robust_valid_csv = csv.writer(csvfile4, delimiter='\t')

                buggy_test_csv = csv.writer(csvfile5, delimiter='\t')
                robust_test_csv = csv.writer(csvfile6, delimiter='\t')



                for (i,row) in enumerate(traces_csv):
                  if (len(row) <> 3) or (not ("/" in row[0])):
                    continue

                  if i in train_vuln_traces:
                    times = train_vuln_traces[i]
                    for _ in range(times):
                      train_csv.writerow([row[0],row[1],0])
                  elif i in train_norm_traces:
                    times = train_norm_traces[i]
                    for _ in range(times):            
                      train_csv.writerow([row[0],row[1],1])
                  elif i in valid_vuln_traces:
                    times = 1
                    #assert(times == 1)
                    for _ in range(times):
                      buggy_valid_csv.writerow(([row[0],row[1],0])) 
                  elif i in valid_norm_traces:
                    times = 1
                    #assert(times == 1)
                    for _ in range(times):
                      robust_valid_csv.writerow(([row[0],row[1],1]))
                  elif i in test_vuln_traces:
                    times = test_vuln_traces[i]
                    assert(times == 1)
                    for _ in range(times):
                      buggy_test_csv.writerow(([row[0],row[1],0])) 
                  elif i in test_norm_traces:
                    times = test_norm_traces[i]
                    assert(times == 1)
                    for _ in range(times):
                      robust_test_csv.writerow(([row[0],row[1],1]))
 
    gc.collect()

    #os.system("echo "+ str(seed) + " > " + seed_filename)
    os.system("zcat "+ tmp_filename + " | shuf | gzip -9 > "+train_filename)
    os.system("rm -f "+cdata_dir+"/model.bin " + tmp_filename)

    sizes_filename = os.path.join(cdata_dir,"sizes.pkl")
    sizes = dict()

    sizes[train_filename] = file_len(train_filename)
    sizes[buggy_valid_filename] = file_len(buggy_valid_filename)
    sizes[robust_valid_filename] = file_len(robust_valid_filename)
    sizes[buggy_test_filename] = file_len(buggy_test_filename)
    sizes[robust_test_filename] = file_len(robust_test_filename)
    
    cPickle.dump(sizes,open(sizes_filename,"w+")) 

dump(train_vuln_programs, train_norm_programs, test_vuln_programs, test_norm_programs)
