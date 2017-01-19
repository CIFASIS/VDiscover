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

import pickle

from Utils import *
from Pipeline import *
from sklearn.metrics import confusion_matrix


def TrainScikitLearn(model_file, train_file, valid_file, vtype, ftype, nsamples):

    modelfile = openModel(model_file)
    train_programs, train_features, train_classes = readTraces(
        train_file, nsamples, cut=None)
    print "using", len(train_features), "examples to train."

    train_dict = dict()
    train_dict[ftype] = train_features

    print "Transforming data and fitting model.."

    if vtype == "bow":
        model = makeTrainPipelineBOW(ftype)

    model.fit(train_dict, train_classes)

    print "Done!"
    # print model
    # print confusion_matrix(train_classes, model.predict(train_dict))

    print "Saving model to", model_file
    modelfile.write(pickle.dumps(model))


"""
def TrainKeras(model_file, train_file, valid_file, ftype, nsamples):

    csvreader = open_csv(train_file)
    modelfile = open_model(model_file)

    train_features = []
    train_programs = []
    train_classes = []

    print "Reading and sampling data to train..",
    if nsamples is None:
        for i, (program, features, cl) in enumerate(csvreader):
            train_programs.append(program)
            train_features.append(features)
            train_classes.append(int(cl))
    else:

        train_size = file_len(in_file)
        skip_until = random.randint(0, train_size - nsamples)

        for i, (program, features, cl) in enumerate(csvreader):

            if i < skip_until:
                continue
            elif i - skip_until == nsamples:
                break

            train_programs.append(program)
            train_features.append(features)
            train_classes.append(int(cl))
    train_size = len(train_features)

    assert(train_size == len(train_classes))

    print "using", train_size, "examples to train."

    train_dict = dict()
    train_dict[ftype] = train_features
    batch_size = 16
    window_size = 25

    from keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(nb_words=None, filters="", lower=False, split=" ")
    # print type(train_features[0])
    tokenizer.fit_on_texts(train_features)
    max_features = len(tokenizer.word_counts)

    preprocessor = KerasPreprocessor(tokenizer, window_size, batch_size)

    if valid_file is not None:
        csvreader = open_csv(valid_file)

        valid_features = []
        valid_programs = []
        valid_classes = []

        print "Reading data to valid..",
        for i, (program, features, cl) in enumerate(csvreader):
            valid_programs.append(program)
            valid_features.append(features)
            valid_classes.append(int(cl))

        print "using", len(train_features), "examples to valid."
        #X_valid,y_valid = preprocessor.preprocess(valid_features, valid_classes)
    else:
        valid_features, train_features = train_features[
            0:int(0.1 * train_size)], train_features[int(0.1 * train_size):]
        valid_classes, train_classes = train_classes[
            0:int(0.1 * train_size)], train_classes[int(0.1 * train_size):]

    X_valid, y_valid = preprocessor.preprocess(
        valid_features, valid_classes, 500)
    X_train, y_train = preprocessor.preprocess(
        train_features, train_classes, 10000)

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
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
               epsilon=1e-8, kappa=1 - 1e-8)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, class_mode="binary")
    #model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=30, validation_data = (X_valid,y_valid), show_accuracy=True)
    model.fit(X_train, y_train, batch_size=batch_size,
              nb_epoch=5, show_accuracy=True)

    print "Saving model to", model_file

    modelfile.write(pickle.dumps(KerasPredictor(preprocessor, model, ftype)))
"""

def Train(model_file, train_file, valid_file, model_type, vector_type, feature_type, nsamples):

    TrainScikitLearn(model_file, train_file, valid_file, vector_type, feature_type, nsamples)

    #elif ttype == "lstm":
    #    try:
    #        import keras
    #    except:
    #        print "Failed to import keras modules to perform LSTM training"
    #        return
    #    TrainKeras(model_file, train_file, valid_file, ftype, nsamples)
