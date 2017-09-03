from Utils import readTraces
from sys import argv
from random import gauss

from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


import matplotlib as mpl
mpl.use('Pdf')
import matplotlib.pyplot as plt

def plot(data, labels, filename):

    cmap_bold = ['#FF0000', '#00FF00', '#0000FF', '#FFFFFF']
    ran0 = sum(data[:,0]) / len(labels) * 0.1
    ran1 = sum(data[:,1]) / len(labels) * 0.1

    plt.clf()
    for i,c in enumerate(labels):
        if c == 0:
            continue
        plt.scatter([data[i,0] + gauss(0.0,ran0)], [data[i,1] + gauss(0.0,ran1)], edgecolor="black", c=[cmap_bold[c]], label=c)

    for i,c in enumerate(labels):
        if c == 1:
            continue
        plt.scatter([data[i,0] + gauss(0.0,ran0)], [data[i,1] + gauss(0.0,ran1)], edgecolor="black", c=[cmap_bold[c]], label=c)



    plt.savefig(filename)

def mytokenizer(s):
    return filter(lambda x: x != '', s.split(" "))

# TFIdf
vectorizer = TfidfVectorizer(tokenizer=mytokenizer, lowercase=False)

print "Reading traces"
progs, traces, cl = readTraces(argv[1], None)

print "Vectorizing.."
vectorizer.fit(traces)
#print vectorizer.get_feature_names()
data = vectorizer.transform(traces)

reducer = TruncatedSVD()
print "Reducing.."
rdata = reducer.fit_transform(data)

plot(rdata, cl, "static-tfidf-svd.pdf")

del rdata, reducer

reducer = make_pipeline(StandardScaler(with_std=False), PCA())
print "Reducing.."
rdata = reducer.fit_transform(data.toarray())

plot(rdata, cl, "static-tfidf-pca.pdf")

#km = KMeans(n_clusters=2, verbose=True)
#print "Clustering.."
#km.fit(data)

#plot(rdata, km.labels_, "vdiscovery-tfidf-pca-kmeans.pdf")

del vectorizer, reducer, data, rdata

# CountVectorizer
vectorizer = CountVectorizer(min_df=2, max_df=128, tokenizer=mytokenizer, lowercase=False)

print "Vectorizing.."
vectorizer.fit(traces)
data = vectorizer.transform(traces)

reducer = TruncatedSVD()
print "Reducing.."
rdata = reducer.fit_transform(data)

plot(rdata, cl, "static-count-svd.pdf")

del rdata, reducer

reducer = make_pipeline(StandardScaler(with_std=False), PCA())
print "Reducing.."
rdata = reducer.fit_transform(data.toarray())

plot(rdata, cl, "static-count-pca.pdf")
