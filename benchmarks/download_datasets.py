#!/usr/bin/env python

import h5py
import numpy as np
import os
import urllib.request

def download_bremen_small():
    print('Downloading Bremen small... ', end='')
    urllib.request.urlretrieve('https://b2share.eudat.eu/api/files/189c8eaf-d596-462b-8a07-93b5922c4a9f/bremenSmall.h5.h5', 'bremen_small.h5')
    print('[OK]')

def download_iris():
    print('Downloading Iris... ', end='')
    request = urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 'iris.csv')
    data = np.genfromtxt('iris.csv', delimiter=',')[:,:-2].astype(np.float32)

    with h5py.File('iris.h5', 'w') as handle:
        handle['DBSCAN'] = data
    os.remove('iris.csv')
    print('[OK]')

def download_twitter_small():
    print('Downloading Twitter small... ', end='')
    urllib.request.urlretrieve('https://b2share.eudat.eu/api/files/189c8eaf-d596-462b-8a07-93b5922c4a9f/twitterSmall.h5.h5', 'twitter_small.h5')
    print('[OK]')

if __name__ == '__main__':
    download_bremen_small()
    download_iris()
    download_twitter_small()

