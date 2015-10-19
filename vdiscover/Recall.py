import gzip
import sys
import csv
import pickle
import numpy

from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, accuracy_score, roc_auc_score, f1_score, recall_score

from Utils import *

def Recall(model_file, in_file, in_type, out_file, test_mode, probability=False):

  model = load_model(model_file)
  csvwriter = write_csv(out_file)

  x = dict()

  testcases, features, test_classes = read_traces(in_file, None, cut=None)
  x[in_type] = features

  if probability:
    predicted_classes = map(lambda x: x[1], model.predict_proba(x)) # probability of the second class
  else:
    predicted_classes = model.predict(x)

  for testcase,y in zip(testcases,predicted_classes):
    csvwriter.writerow([testcase,y])

  if test_mode == "simple":
    nclasses = len(set(test_classes))

    if nclasses == 1:
      err = [None, None]
      err[test_classes[0]] = recall_score(test_classes, predicted_classes, average=None)[0]
      err[1 - test_classes[0]] = 1.0
    else:
      err = recall_score(test_classes, predicted_classes, average=None)

    print classification_report(test_classes, predicted_classes)
    print "Accuracy per class:", round(err[0],2), round(err[1],2)
    print "Average accuracy:", round(sum(err)/2.0,2)

  elif test_mode == "aggregated":


    #print len(testcases), len(predicted_classes), len(test_classes)
    prog_pred = dict()

    for (program, predicted, real) in zip(testcases, predicted_classes, test_classes):
      prog_pred[program] = prog_pred.get(program,[]) + [abs(predicted-real)]

    print round(numpy.mean(map(numpy.mean, prog_pred.values())),2)

    # BROKEN!
    #prog_classes = dict()
    #for prog,cl in zip(testcases, test_classes):
    #  prog_classes[prog] = cl

    #prog_pred = dict(zip(prog_classes.keys(), [[]]*len(prog_classes)))
    #for prog, pred in zip(testcases,predicted_classes):
    #  prog_pred[prog].append(abs(pred - prog_classes[prog]))

    #errors = []
    #for prog, preds in prog_pred.items():
    #  errors.append(sum(preds)/float(len(preds)))

    #print sum(errors) / float(len(errors))


