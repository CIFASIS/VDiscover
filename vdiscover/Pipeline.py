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

from sklearn.base import  BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from random import randint, sample

def static_tokenizer(s):
    return filter(lambda x: x<>'', s.split(" "))

def dynamic_tokenizer(s):
    return filter(lambda x: x<>'', s.split(" "))

class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

def make_pipeline(ftype):
  
  if ftype is "dynamic":
    return Pipeline(steps=[
         ('selector', ItemSelector(key='dynamic')),
         ('dvectorizer', CountVectorizer(tokenizer=static_tokenizer, ngram_range=(2,2), lowercase=False)),
         ('todense', DenseTransformer()),
         ('classifier', RandomForestClassifier(n_estimators=1000, max_features=None, max_depth=100, class_weight="auto"))
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

try:
  from keras.preprocessing import sequence
except:
  pass

 
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

      i = randint(0, X_size-1)
      
      raw_trace = X_data[i]
      trace = raw_trace.split(" ") 

      size = len(trace)

      start = randint(0, size-1)
      end = start + randint(0, self.max_len)

      new_trace = " ".join(trace[start:(end+1)])
      cut_X_data.append(new_trace)

      if y_data is not None:
        y = y_data[i]
        cut_y_data.append(y)

    X_train = self.tokenizer.texts_to_sequences(cut_X_data)
    y_train = cut_y_data
 
    if y_train is not None:
      X_train,y_train = zip(*filter(lambda (x,y): not (x == []), zip(X_train,y_train)))
    else:
      X_train = filter(lambda x: not (x == []), X_train)


    X_size = len(X_train)
    X_train = X_train[:(X_size-(X_size % self.batch_size))] 
    X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
   
    if y_train is not None:
      y_train = y_train[:(X_size-(X_size % self.batch_size))]
      return X_train,y_train
    else:
      return X_train


  def preprocess_one(self, raw_trace, sample_size=100):

    trace = raw_trace.split(" ") 
    size = len(trace)
    cut_X_data = []
    #print trace

    for _ in xrange(sample_size):

      start = randint(0, size-1)
      end = start + randint(0, self.max_len)

      new_trace = " ".join(trace[start:(end+1)])
      cut_X_data.append(new_trace)

    X_train = self.tokenizer.texts_to_sequences(cut_X_data)
    X_train = filter(lambda x: not (x == []), X_train)

    X_size = len(X_train)
    X_train = X_train[:(X_size-(X_size % self.batch_size))]
    #print "X_size", X_size-(X_size % self.batch_size) 
 
    X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
    return X_train

class KerasPredictor:

    def __init__(self,preprocessor, model, ftype):
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
          predictions = self.model.predict(trace_data, verbose=0, batch_size=self.batch_size)
        else: # imposible to predict
          predictions = [0]

        avg_predictions = sum(predictions)/100.0
        #print predictions, avg_predictions
        if avg_predictions > 0.5:
          X_predictions.append(1)
        else:
          X_predictions.append(0)

      return X_predictions
