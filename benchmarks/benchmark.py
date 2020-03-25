#!/usr/bin/env python

import numpy as np
import subprocess
import sys
import time

DATASET_PARAMETERS = {# eps,  min_points
    #'bremen_small.h5':  (100,  312),
    'iris.h5':          (0.32,  3),
    #'twitter_small.h5': (0.01, 40),
}
TRIALS = 10

def run_benchmark(command, log_path):
    sys_stdout, sys_stderr = sys.stdout, sys.stderr
    log_handle = open(log_path, 'w')
    sys.stdout, sys.stderr = log_handle, log_handle

    for dataset, parameters in DATASET_PARAMETERS.items():
        eps, min_points = parameters

        timings = np.empty((TRIALS,))
        print('Running benchmarks for', dataset)
        for i in range(TRIALS):
            start = time.perf_counter()
            subprocess.run(command.format(dataset=dataset, eps=eps, min_points=min_points), shell=True)
            end = time.perf_counter()
            timings[i] = end - start
            print('\t', i, timings[i])

        print('Average:', timings.mean(), ' Deviation:', timings.std())
        print('')

    sys.stdout, sys.stderr = sys_stdout, sys_stderr


if __name__ == '__main__':
    run_benchmark('./sklearn-dbscan.py {dataset} -e {eps} -m {min_points}', 'sklearn.log')
    run_benchmark('../build/hpdbscan -i {dataset} --input-dataset DBSCAN -o output.h5 --output-dataset CLUSTERS -e {eps} -m {min_points}', 'hpdbscan.log')

