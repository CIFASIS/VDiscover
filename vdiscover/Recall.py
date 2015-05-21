import gzip
import sys
import csv
import pickle

csv.field_size_limit(sys.maxsize)

def Recall(model_file, in_file, in_type, out_file):

  if ".gz" in in_file:
    infile = gzip.open(in_file, "r")
  else:
    infile = open(in_file, "r")
 
  csvreader = csv.reader(infile, delimiter='\t')
 
  outfile = open(out_file, "a+")
  csvwriter = csv.writer(outfile, delimiter='\t')

  model = pickle.load(gzip.open(model_file))

  for prog, features,_ in csvreader:
    x = dict()
    x[in_type] = [features]
    csvwriter.writerow([prog, model.predict(x)[0]])

