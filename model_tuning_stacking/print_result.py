import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='print result')
parser.add_argument('--dataset', type=str, required=True,
                    help='result for best model')
parser.add_argument('--save-dir',type=str, required=True,
                    help='')
args = parser.parse_args()

data = pd.read_csv(args.dataset)

def line_parse(line):
    result=' '.join([str(k) for k, v in enumerate(line[1:]) if v == 1])
    result= str(line[0])+',' + result
    return result

with open(args.save_dir, 'w') as f:
    f.write('business_id')
    f.write(',')
    f.write('labels')
    f.write('\n')
    for index, row in data.iterrows():
        f.write(line_parse(row))
        f.write('\n')






