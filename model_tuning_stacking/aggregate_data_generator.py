import pandas as pd
import numpy as np
import os
import argparse
from numpy import mean
from numpy import max
from numpy import median
import re
import itertools
from subroutine import path_generator
parser = argparse.ArgumentParser(description='cleaning and pooling w.r.t business_id, results will be save in current dir')
parser.add_argument('--data-dir', nargs='+', type=str, default='pooling_data2',
                    help='the input data directory')
parser.add_argument('--dataset', nargs='+', type=str, required=True,
                    help='dataset name')
args = parser.parse_args()


meta_data = []
for value in args.data_dir:
    path_tuple = list(itertools.product(value, args.dataset))
    paths = path_generator(path_tuple)
    data_list = []
    for key, value in enumerate(paths):
        data_list[k] = pd.read_csv(value)
    current = data_list[0]
    for i, frame in enumerate(data_list[1:], 2):
            current = current.merge(frame, on='business_id')
    if value == pooling_data2:
        names = current.columns[1:]
        names = ['mean'+'_' + k for k in names]
        current.columns = ['business_id'] + names










