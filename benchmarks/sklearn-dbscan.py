#!/usr/bin/env python

import argparse
import h5py
import numpy as np
import sklearn.cluster

def cluster(arguments):
    with h5py.File(arguments.file, 'r') as handle:
        data = np.array(handle['DBSCAN'])

    dbscan = sklearn.cluster.DBSCAN(eps=arguments.e, min_samples=arguments.m)
    dbscan.fit(data)

    with h5py.File('output.h5', 'w') as handle:
       handle['Clusters'] = dbscan.labels_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=float, help='spatial search radius epsilon')
    parser.add_argument('-m', type=int, help='density threshold min_points')
    parser.add_argument('file', type=str, help='file to cluster')
    args = parser.parse_args()

    cluster(args)

