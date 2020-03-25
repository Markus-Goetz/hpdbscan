#!/usr/bin/env python

import glob
import pandas as pd
import re
import seaborn as sns

sns.set(style="whitegrid")

benchmark_name_pattern = re.compile('Running benchmarks for ([a-zA-Z0-9_]+)\.h5')
timings_pattern = re.compile('	 [0-9]+ ([0-9]+\.[0-9]+)')

def find_logs():
    return glob.glob('./*.log')

def parse_logs(logs):
    data = pd.DataFrame(columns=['benchmark', 'execution time in s', 'tool'])
    offset = 0

    for log in logs:
        tool = log[2:-4]
            
        with open(log, 'r') as handle:
            content = handle.read()
            benchmarks = benchmark_name_pattern.findall(content)
            timings = timings_pattern.findall(content)
            
            for i, b in enumerate(benchmarks):
                for j in range(10):                
                    data.loc[i + offset] = [b, float(timings[i * 10 + j]), tool]
                    offset += 1
            
    return data
            
def plot(data):
    bar_plot = sns.barplot(
        x='benchmark', 
        y='execution time in s', 
        hue='tool', 
        data=data
    )
    bar_plot.get_figure().savefig('benchmark.png')
    

if __name__ == '__main__':
    logs = find_logs()
    data = parse_logs(logs)
    plot(data)

