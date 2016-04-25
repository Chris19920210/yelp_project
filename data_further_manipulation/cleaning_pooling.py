import pandas as pd
import numpy as np
import os
import argparse
from numpy import mean
from numpy import max
import re


parser = argparse.ArgumentParser(description='cleaning and pooling w.r.t business_id, results will be save in current dir')
parser.add_argument('--match-file', type=str, required=True,
                    help='load photo_to_biz_ids.csv')
parser.add_argument('--data-dir', type=str, default='npy_data',
                    help='the input data directory')
parser.add_argument('--dataset', type=str, required=True,
                    help='dataset name')
parser.add_argument('--pooling-method', type=str, default='max',
                    choices=['max', 'mean'],
                    help='pooling method for aggregate feature')
args = parser.parse_args()

# load data to pandas
data_path = os.path.join(args.data_dir, args.dataset)
X = np.load(data_path)
X = pd.DataFrame(X)

# photo_id to business id
photo2biz = pd.read_csv(args.match_file)
# generate the dict key = business_id, value = photo_id
dicts = photo2biz.copy().groupby("business_id")
# reconstrunct the index
photo2biz.sort_values("photo_id", inplace=True)

# drop duplicates
photo2biz.drop_duplicates("photo_id", inplace=True)
# get the photo_id to merge X
Id = pd.DataFrame(photo2biz["photo_id"])
Id.reset_index(drop=True, inplace=True)
# merge
X = pd.concat([Id, X], axis=1)
# set Id as index
X.set_index('photo_id', inplace=True)
# possible methods for pooling
methods = {'max': max, 'mean': mean}
# pooling based on business id
def main():
    tmp = []
    for k, v in dicts:
        # get photo_id for specific business_id
        index = v['photo_id']
        # retrieve the data
        batch = X.loc[index, :]
        # aggregate data
        agg_featrue = batch.apply(methods[args.pooling_method], axis=0)
        line = [k] + list(agg_featrue)
        tmp.append(line)
    #create data frame for convenience
    results = pd.DataFrame(tmp)
    h, w = results.shape
    # change the column name
    name = args.dataset[:args.dataset.rfind('_')] + '_' + re.findall(r'\d+', args.dataset)[0]
    results.columns = ['business_id'] + [name + '_' + str(i) for i in range(1, w)]
    return results

if __name__ == '__main__':
    final_result = main()
    final_result.to_csv('./' + args.dataset[:args.dataset.find('.')] + '.csv', index=False)









