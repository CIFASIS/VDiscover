import gzip
import sys
import csv
import pickle

from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, accuracy_score, roc_auc_score, f1_score, recall_score

csv.field_size_limit(sys.maxsize)

def Recall(model_file, in_file, in_type, out_file, probability=False, test=False):

  testcases,test_classes = [],[]

  if ".gz" in in_file:
    infile = gzip.open(in_file, "r")
  else:
    infile = open(in_file, "r")
 
  csvreader = csv.reader(infile, delimiter='\t')
 
  outfile = open(out_file, "a+")
  csvwriter = csv.writer(outfile, delimiter='\t')

  model = pickle.load(gzip.open(model_file))

  x = dict()
  x[in_type] = []

  for row in csvreader:
    testcase, features = row[0], row[1]
    testcases.append(testcase)

    if test:
      test_classes.append(int(row[2]))

    x[in_type].append(features)

  if probability:
    predicted_classes = map(lambda x: x[1], model.predict_proba(x)) # probability of the second class
  else:
    predicted_classes = model.predict(x)
  
  #print predicted_classes

  for testcase,y in zip(testcases,predicted_classes):
    csvwriter.writerow([testcase,y])

  if test:
    #print confusion_matrix(test_classes, predicted_classes)
    nclasses = len(set(test_classes))

    if nclasses == 1:
      err = [None, None]
      err[test_classes[0]] = recall_score(test_classes, predicted_classes, average=None)[0]
      err[1 - test_classes[0]] = 1.0
    else: 
      err = recall_score(test_classes, predicted_classes, average=None)
    #print err
    print err[0], err[1], sum(err)/2.0
    #print classification_report(test_classes, predicted_classes)


