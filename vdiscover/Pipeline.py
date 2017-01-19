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
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
#from sklearn.manifold import MDS

from random import random, randint, sample, gauss, shuffle


def staticTokenizer(s):
    return filter(lambda x: x != '', s.split(" "))


def dynamicTokenizer(s):
    return filter(lambda x: x != '', s.split(" "))


class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return []


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

    def get_params(self, deep=True):
        return []


class CutoffMax(BaseEstimator, TransformerMixin):

    def __init__(self, maxv):
        self.maxv = maxv

    def fit(self, x, y=None):
        #self.pos = x > self.maxv
        return self

    def transform(self, X, y=None, **fit_params):
        self.pos = X > self.maxv
        X[self.pos] = self.maxv
        return X

    def get_params(self, deep=True):
        return []


def makeTrainPipelineBOW(ftype):

    if ftype is "dynamic":

        realpath = os.path.dirname(os.path.realpath(__file__))
        f = open(realpath + "/data/dyn_events.dic")

        event_dict = []

        for line in f.readlines():
            event_dict.append(line.replace("\n", ""))

        return Pipeline(steps=[
            ('selector', ItemSelector(key='dynamic')),
            ('dvectorizer', CountVectorizer(tokenizer=dynamicTokenizer,
                                            ngram_range=(1, 3), lowercase=False, vocabulary=event_dict)),
            ('todense', DenseTransformer()),
            ('cutfoff', CutoffMax(16)),
            ('classifier', RandomForestClassifier(
                n_estimators=1000, max_features=None, max_depth=100))
            #('classifier',  GaussianNB())

        ])
    elif ftype is "static":
        return Pipeline(steps=[
            ('selector', ItemSelector(key='static')),
            ('dvectorizer', CountVectorizer(
                tokenizer=static_tokenizer, ngram_range=(1, 1), lowercase=False)),
            ('todense', DenseTransformer()),
            ('classifier', LogisticRegression(penalty="l2", C=1e-07, tol=1e-06))
        ])
    else:
        assert(0)


def makeClusterPipelineBOW(ftype, rdim):
    if ftype is "dynamic" and rdim == "pca":

        return Pipeline(steps=[
            ('selector', ItemSelector(key='dynamic')),
            ('dvectorizer', TfidfVectorizer(tokenizer=dynamicTokenizer,
                                            use_idf=False, norm=None, ngram_range=(1, 1), lowercase=False)),
            ('todense', DenseTransformer()),
            #('cutfoff', CutoffMax(16)),
            ('reducer', PCA(n_components=2)),

        ])

    elif ftype is "dynamic" and rdim == "svd":

        return Pipeline(steps=[
            ('selector', ItemSelector(key='dynamic')),
            ('dvectorizer', TfidfVectorizer(tokenizer=dynamicTokenizer,
                                            use_idf=False, norm=None, ngram_range=(1, 1), lowercase=False)),
            ('todense', DenseTransformer()),
            #('cutfoff', CutoffMax(16)),
            ('reducer', TruncatedSVD(n_components=2)),

        ])

    elif ftype is "dynamic" and rdim == "none":

        return Pipeline(steps=[
            ('selector', ItemSelector(key='dynamic')),
            ('dvectorizer', TfidfVectorizer(tokenizer=dynamic_tokenizer,
                                            use_idf=False, norm=None, ngram_range=(1, 1), lowercase=False)),
            ('todense', DenseTransformer()),
            #('cutfoff', CutoffMax(16)),
        ])

    elif ftype is "static":
        return Pipeline(steps=[
            ('selector', ItemSelector(key='static')),
            ('dvectorizer', TfidfVectorizer(tokenizer=dynamic_tokenizer,
                                            use_idf=False, norm=None, ngram_range=(1, 1), lowercase=False)),
            ('todense', DenseTransformer()),
            ('cutfoff', CutoffMax(16)),
            ('reducer', PCA(n_components=2)),
        ])
    else:
        assert(0)


def makeClusterPipelineDoc2vec(ftype, rdim):
    if ftype is "dynamic" and rdim == "pca":
        return Pipeline(steps=[
            ('selector', ItemSelector(key='dynamic')),
            ('reducer', PCA(n_components=2)),
        ])
    elif ftype is "dynamic" and rdim == "svd":
        return Pipeline(steps=[
            ('selector', ItemSelector(key='dynamic')),
            ('reducer', TruncatedSVD(n_components=2)),
        ])
    elif ftype is "dynamic" and rdim == "none":
        return Pipeline(steps=[
            ('selector', ItemSelector(key='dynamic'))
        ])
    elif ftype is "static":
        raise NotImplemented
    else:
        assert(0)


def makeClusterPipelineSubtraces(ftype):
    if ftype is "dynamic":
        return Pipeline(steps=[
            ('selector', ItemSelector(key='dynamic')),
            ('reducer', PCA(n_components=12)),
        ])
    elif ftype is "static":
        raise NotImplemented
    else:
        assert(0)

"""
def make_cluster_cnn(
        mode,
        max_features,
        maxlen,
        embedding_dims,
        nb_filters,
        filter_length,
        hidden_dims,
        nb_classes,
        weights=None):

    # print mode, max_features, maxlen, embedding_dims, nb_filters,
    # filter_length, hidden_dims, nb_classes
    from keras.preprocessing import sequence
    from keras.optimizers import RMSprop
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.embeddings import Embedding
    from keras.layers.convolutional import Convolution1D, MaxPooling1D

    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    if mode == "train":
        model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    elif mode == "test":
        model.add(Embedding(max_features, embedding_dims,
                            input_length=maxlen, weights=weights[0]))

    model.add(Dropout(0.25))

    # we add a Convolution1D, which will learn nb_filters
    # word group filters of size filter_length:
    if mode == "train":
        model.add(Convolution1D(nb_filter=nb_filters,
                                filter_length=filter_length,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1))

    elif mode == "test":
        model.add(Convolution1D(nb_filter=nb_filters,
                                filter_length=filter_length,
                                border_mode='valid',
                                activation='relu',
                                subsample_length=1,
                                weights=weights[2]))

    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling1D(pool_length=2))

    # We flatten the output of the conv layer, so that we can add a vanilla
    # dense layer:
    model.add(Flatten())

    # Computing the output shape of a conv layer can be tricky;
    # for a good tutorial, see: http://cs231n.github.io/convolutional-networks/
    output_size = nb_filters * (((maxlen - filter_length) / 1) + 1) / 2
    # print output_size, hidden_dims

    # We add a vanilla hidden layer:
    if mode == "train":
        model.add(Dense(hidden_dims))
    if mode == "test":
        model.add(Dense(hidden_dims, weights=weights[5]))

    if mode == "train":

        model.add(Dropout(0.25))
        model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a
        # sigmoid:
        model.add(Dense(nb_classes))

        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', class_mode="categorical")

    elif mode == "test":
        model.compile(loss='mean_squared_error', optimizer='rmsprop')

    return model

try:
    from keras.preprocessing import sequence
except:
    pass

class DeepReprPreprocessor:

    def __init__(self, tokenizer, max_len, batch_size):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size

    def preprocess_traces(self, X_data, y_data=None, labels=None):

        cut_X_data = []
        cut_label_data = []
        cut_y_data = []
        #rep = 5

        X_size = len(X_data)

        for i, x in enumerate(X_data):

            #i = randint(0, X_size-1)

            raw_trace = x[:-1]
            trace = raw_trace.split(" ")

            size = len(trace)
            rep = 1 + int(float(size) / float(self.max_len))
            rep = min(rep, 10)

            for _ in range(rep):

                start = size - (self.max_len)
                start = randint(0, max(start, 0))

                new_trace = " ".join(trace[start:(start + self.max_len)])
                # print "sizes:", size, len(trace[start:(start+self.max_len)])

                cut_X_data.append(new_trace)

                if labels is not None:
                    cut_label_data.append(labels[i])
                else:
                    cut_label_data.append("+" + str(size))

                if y_data is not None:
                    cut_y_data.append(y_data[i])
                else:
                    cut_y_data.append(0)

        X_train = self.tokenizer.texts_to_sequences(cut_X_data)
        labels = cut_label_data
        y_train = cut_y_data
        X_train, y_train, labels = zip(
            *filter(lambda x_y_z: not (x_y_z[0] == []), zip(X_train, y_train, labels)))

        X_size = len(X_train)
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        return X_train, y_train, labels

    def preprocess(self, X_data, cut_size=1):

        cut_X_data = []
        cut_y_data = []
        self.classes = []
        X_size = len(X_data)
        stats = dict()

        for _ in xrange(1000):

            i = randint(0, X_size - 1)

            raw_trace = X_data[i][:-1]
            trace = raw_trace.split(" ")

            size = len(trace)

            if size <= (self.max_len + 1):
                start = 0
                end = size - 2
                new_trace = " ".join(trace[start:(end + 1)])
                last_event = trace[(end + 1)].split(":")
                cut_y_data.append(last_event[0])
            else:
                # print size
                start = size - (self.max_len) - 2
                start = randint(0, start)
                end = start + self.max_len
                # print len(trace[start:end])
                #new_trace = " ".join(trace[start:end])

                #start = randint(0, size-2)
                #end = randint(start, size-2)

                new_trace = " ".join(trace[start:(end + 1)])
                last_event = trace[end + 1].split(":")
                cut_y_data.append(last_event[0])

        for y in set(cut_y_data):
            stats[y] = float(cut_y_data.count(y)) / len(cut_y_data)

        print stats, sum(stats.values())
        # assert(0)
        cut_y_data = []
        for _ in xrange(cut_size):

            i = randint(0, X_size - 1)

            raw_trace = X_data[i][:-1]
            trace = raw_trace.split(" ")
            size = len(trace)

            if size <= (self.max_len + 1):
                start = 0
                end = size - 2
                new_trace = " ".join(trace[start:(end + 1)])
                last_event = trace[(end + 1)].split(":")
            else:
                # print size
                start = size - (self.max_len) - 2
                start = randint(0, start)
                end = start + self.max_len
                # print len(trace[start:end])
                #new_trace = " ".join(trace[start:end])

                #start = randint(0, size-2)
                #end = randint(start, size-2)

                new_trace = " ".join(trace[start:(end + 1)])
                last_event = trace[end + 1].split(":")

            cl = last_event[0]

            if cl not in self.classes:
                self.classes.append(cl)
                stats[cl] = 0.0
            else:
                if random() <= stats[cl]:
                    continue

            cut_X_data.append(new_trace)
            cut_y_data.append(self.classes.index(cl))

        X_train = self.tokenizer.texts_to_sequences(cut_X_data)

        y_train = []

        for y in cut_y_data:
            v = [0] * len(self.classes)
            v[y] = 1
            y_train.append(v)

        X_train = filter(lambda x: not (x == []), X_train)

        X_size = len(X_train)
        X_train = X_train[:(X_size - (X_size % self.batch_size))]
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)

        if y_train is not None:
            y_train = y_train[:(X_size - (X_size % self.batch_size))]
            return X_train, y_train
        else:
            return X_train


class KerasPreprocessor:

    def __init__(self, tokenizer, max_len, batch_size):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size

    def preprocess(self, X_data, y_data=None, cut_size=1):

        cut_X_data = []
        cut_y_data = []
        X_size = len(X_data)

        for _ in xrange(cut_size):

            i = randint(0, X_size - 1)

            raw_trace = X_data[i]
            trace = raw_trace.split(" ")

            size = len(trace)

            start = randint(0, size - 1)
            end = start + randint(0, self.max_len)

            new_trace = " ".join(trace[start:(end + 1)])
            cut_X_data.append(new_trace)

            if y_data is not None:
                y = y_data[i]
                cut_y_data.append(y)

        X_train = self.tokenizer.texts_to_sequences(cut_X_data)
        y_train = cut_y_data

        if y_train is not None:
            X_train, y_train = zip(
                *filter(lambda x_y: not (x_y[0] == []), zip(X_train, y_train)))
        else:
            X_train = filter(lambda x: not (x == []), X_train)

        X_size = len(X_train)
        X_train = X_train[:(X_size - (X_size % self.batch_size))]
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)

        if y_train is not None:
            y_train = y_train[:(X_size - (X_size % self.batch_size))]
            return X_train, y_train
        else:
            return X_train

    def preprocess_one(self, raw_trace, sample_size=100):

        trace = raw_trace.split(" ")
        size = len(trace)
        cut_X_data = []
        # print trace

        for _ in xrange(sample_size):

            start = randint(0, size - 1)
            end = start + randint(0, self.max_len)

            new_trace = " ".join(trace[start:(end + 1)])
            cut_X_data.append(new_trace)

        X_train = self.tokenizer.texts_to_sequences(cut_X_data)
        X_train = filter(lambda x: not (x == []), X_train)

        X_size = len(X_train)
        X_train = X_train[:(X_size - (X_size % self.batch_size))]
        # print "X_size", X_size-(X_size % self.batch_size)

        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        return X_train


class KerasPredictor:

    def __init__(self, preprocessor, model, ftype):
        self.preprocessor = preprocessor
        self.batch_size = preprocessor.batch_size
        self.ftype = ftype
        self.model = model

    def predict(self, X_data):
        X_size = len(X_data)
        X_data = X_data[self.ftype]
        X_predictions = []

        for raw_trace in X_data:

            trace_data = self.preprocessor.preprocess_one(raw_trace)

            if len(trace_data) > 0:
                predictions = self.model.predict(
                    trace_data, verbose=0, batch_size=self.batch_size)
            else:  # imposible to predict
                predictions = [0]

            avg_predictions = sum(predictions) / 100.0
            # print predictions, avg_predictions
            if avg_predictions > 0.5:
                X_predictions.append(1)
            else:
                X_predictions.append(0)

        return X_predictions

"""
