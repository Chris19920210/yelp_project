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
                    default=['pooling_data', 'pooling_data2'],
                    help='the input data directory')
parser.add_argument('--dataset', nargs='+', type=str,
                    default=['Inception-7_train384.csv', 'Inception_BN_train224.csv', 'Inception_train224.csv'],
                    help='dataset name')
parser.add_argumnet('--save-dir', type=str, default='./',
                    help='save directory')
args = parser.parse_args()


def main():
    meta_data = []
    for value in args.data_dir:
        path_tuple = list(itertools.product([value], args.dataset))
        paths = path_generator(path_tuple)
        data_list = []
        for key, value in enumerate(paths):
            data_list[k] = pd.read_csv(value)
        current = data_list[0]
        for i, frame in enumerate(data_list[1:]):
            current = current.merge(frame, on='business_id')
        names = current.columns[1:]
        if value == 'pooling_data2':
            names = ['mean'+'_' + k for k in names]
        else:
            names = ['max'+'_' + k for k in names]
        current.columns = ['business_id'] + names
        meta_data.append(current)
    X = meta_data[0].merge(meta_data[1], on='business_id')
    return X


if __name__ == '__main__':
    final_result = main()
    final_result.to_csv(args.save_dir + 'all_data.csv',  index=False)



