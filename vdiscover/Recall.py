import gzip
import sys
import csv
import pickle
import numpy

from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, accuracy_score, roc_auc_score, f1_score, recall_score

from Utils import *


def Recall(
        model_file,
        in_file,
        in_type,
        out_file,
        test_mode,
        probability=False):

    model = loadModel(model_file)
    csvwriter = writeCSV(out_file)

    x = dict()

    testcases, features, test_classes = readTraces(in_file, None, cut=None)
    x[in_type] = features

    if probability:
        # probability of the second class
        predicted_classes = map(lambda x: x[1], model.predict_proba(x))
    else:
        predicted_classes = map(str, model.predict(x))
        #predicted_classes = model.predict(x)

    for testcase, y in zip(testcases, predicted_classes):
        csvwriter.writerow([testcase, y])

    if test_mode == "simple":
        nclasses = len(set(test_classes))
        one_class = int(test_classes[0])

        if nclasses == 1:
            err = [None, None]
            err[one_class] = recall_score(
                test_classes, predicted_classes, average=None)[one_class]
            err[1 - one_class] = err[one_class]
        else:
            err = recall_score(test_classes, predicted_classes, average=None)

        print classification_report(test_classes, predicted_classes)
        print "Accuracy per class:", round(err[0], 2), round(err[1], 2)
        print "Average accuracy:", round(sum(err) / 2.0, 2)

    elif test_mode == "aggregated":

        # print len(testcases), len(predicted_classes), len(test_classes)
        prog_pred = dict()

        for (program, predicted, real) in zip(
             testcases, predicted_classes, test_classes):
            predicted,real = int(predicted), int(real)
            prog_pred[program] = prog_pred.get(
                program, []) + [abs(predicted - real)]

        print round(numpy.mean(map(numpy.mean, prog_pred.values())), 2)
