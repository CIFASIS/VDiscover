import gzip
import sys
import csv
import pickle

from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, accuracy_score, roc_auc_score, f1_score, recall_score

csv.field_size_limit(sys.maxsize)

def Recall(model_file, in_file, in_type, out_file, probability=False, test=False):

  test_classes,predicted_classes = [],[]

  if ".gz" in in_file:
    infile = gzip.open(in_file, "r")
  else:
    infile = open(in_file, "r")
 
  csvreader = csv.reader(infile, delimiter='\t')
 
  outfile = open(out_file, "a+")
  csvwriter = csv.writer(outfile, delimiter='\t')

  model = pickle.load(gzip.open(model_file))

  for row in csvreader:
    prog, features = row[0], row[1]

    x = dict()
    x[in_type] = [features]
    if probability:
      y = [model.predict_proba(x)[0][1]] # probability of the second class
    else:
      y = [model.predict(x)[0]]
    csvwriter.writerow([prog]+y)

    if test:
      test_classes.append(int(row[2]))
      predicted_classes.append(y[0])

  if test:
    #print confusion_matrix(test_classes, predicted_classes)
    nclasses = len(set(test_classes))
    
    if nclasses == 1:
      err = [None, None]
      err[test_classes[0]] = recall_score(test_classes, predicted_classes)
      err[1 - test_classes[0]] = 1.0
    else: 
      err = recall_score(test_classes, predicted_classes, average=None)
    #print err
    print err[0], err[1], sum(err)/2.0
    #print classification_report(test_classes, predicted_classes)


