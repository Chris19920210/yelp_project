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
parser = argparse.ArgumentParser(description='aggregate_data for ')
parser.add_argument('--data-dir', nargs='+', type=str,
                    default=['pooling_data', 'pooling_data2','pooling_data3'],
                    help='the input data directory')
parser.add_argument('--dataset', nargs='+', type=str,
                    default=['Inception-7_train384.csv', 'Inception_BN_train224.csv', 'Inception_train224.csv'],
                    help='dataset name')
parser.add_argument('--save-dir', type=str, default='./',
                    help='save directory')
args = parser.parse_args()

d = {'pooling_data': 'max', 'pooling_data2': 'mean', 'pooling_data3': 'median'}
def main(dicts):
    meta_data = []
    for value in args.data_dir:
        path_tuple = list(itertools.product([value], args.dataset))
        paths = path_generator(path_tuple)
        data_list = []
        for v in paths:
            data_list.append(pd.read_csv(v))
        current = data_list[0]
        for frame in data_list[1:]:
            current = current.merge(frame, on='business_id')
        names = current.columns[1:]
        d = { k: dicts[value] + '_' + k for k in current.columns[1:]}
        d['business_id'] = 'business_id'
        current.rename(columns = d, inplace = True)
        meta_data.append(current)
    X = meta_data[0]
    for value in meta_data[1:]:
        X = X.merge(value, on='business_id')
    return X


if __name__ == '__main__':
    final_result = main(d)
    final_result.to_csv(args.save_dir + 'all_data.csv',  index=False)



