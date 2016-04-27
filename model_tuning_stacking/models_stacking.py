import sys
import math
import numpy as np
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
import xgboost
import os
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import argparse
import re
import collections
from myTransformer import MyTransformer, DataFrameSeparator, DataFrameSelector
from collections import defaultdict
from subroutine import parse,  RetrievalDict, pipeline_generator
from subroutine import PartialStringColumns

# parser for model stacking
parser = argparse.ArgumentParser(description='tuning various models')
parser.add_argument('--labels-file', type=str, required=True,
                    help='labels for supervised learning')
parser.add_argument('--nclass', type=str, required=True,
                    help='choose one out of 9 classes')
parser.add_argument('--data-dir', type=str, default='.',
                    help='the input data directory')
parser.add_argument('--dataset', type=str, required=True,
                    help='dataset name')
parser.add_argument('--top', type=int, required=1,
                    help='top k model for combination')
parser.add_argument('--mean-pooling', nargs='+', type=str, required=True,
                    choices=['mean_Inception-7', 'mean_Inception_BN', 'mean_Inception'],
                    help='mean pooling data set, put cv result in mean_pooling_result')
parser.add_argument('--max-pooling', nargs='+', type=str, required=True,
                    choices=['max_Inception-7', 'max_Inception_BN', 'max_Inception'],
                    help='max pooling data set, put cv result in max_pooling_result')
parser.add_argument('--num-tree', nargs='+', type=int, default=[100, 101, 1],
                    help='# of trees for tree based methods')
parser.add_argument('--depths', nargs='+', type=int, default=[3, 4, 1],
                    help='depths for tree based methods')
parser.add_argument('--lr', nargs='+', type=float, default=[1, 2, 1],
                    help='learning rate for gradient based model')
parser.add_argument('--alpha', nargs='+', type=float, default=[1, 2, 1],
                    help='rate for penalty')
parser.add_argument('--subsample', type=float, default=[1, 2, 1],
                    help='rate for subsample')
parser.add_argument('--colsample-bytree', type=float, default=[0.8, 1.2, 0.2], help='rate for feature bagging')
args = parser.parse_args()

# load data and labels
data_path = os.path.join(args.data_dir, args.dataset)
X = pd.read_csv(data_path)
X.drop('business_id', axis=1, inplace=True)
y = pd.read_csv(args.labels_file)[args.nclass]


# potential methods
methods = {'svm': SGDClassifier(loss='hinge'),
            'msvm': SGDClassifier(loss='squared_hinge'),
            'huber': SGDClassifier(loss='modified_huber'),
            'logistic': SGDClassifier(loss='log'),
            'xgboost': XGBClassifier(),
            'adaboost': AdaBoostClassifier(),
            'extra_tree': ExtraTreesClassifier(),
            'random_forest': RandomForestClassifier(),
            'lda': LinearDiscriminantAnalysis(),
            'qda': QuadraticDiscriminantAnalysis()}

#  data preparation
#  generate metadata list
dict_list = []
# path to mean_pooling_result
path1 = 'mean_pooling_result'
# path to mean_pooling_result
path2 = 'max_pooling_result'

# files that with specific labels
files = [f for f in os.listdir(path1) if re.match(r'Result_for_Class_' + re.escape(args.nclass) + r'.*', f)]

# pattern in order to retrieve file by args
pattern = r"^(Result_for_Class_)\d+_(\w+.*)_\w+\d+\.txt$"


for value in args.mean_pooling + args.max_pooling:
    if re.match(r'mean.*', value):
        # retrieve wanted file
        tmp_list = [f for f in files if re.search(pattern, f).group(2) == value[value.find('_'):][1:]]
        # path to wanted file
        path = os.path.join(path1, tmp_list[0])
        d = RetrievalDict(value, path, args.top, methods)
    else:
        tmp_list = [f for f in files if re.search(pattern, f).group(2) == value[value.find('_'):][1:]]
        path = os.path.join(path2, tmp_list[0])
        d = RetrievalDict(value, path, args.top, methods)
    dict_list.append(d)

# transform to dictionary
agg_dicts = defaultdict(d)
for value in dict_list:
    k, v = value.items()
    agg_dicts[k] = v


# general pipeline
def main():
    clf = Pipeline([
            ('separator', DataFrameSeparator()),
            ('union', FeatureUnion(
                transformer_list=pipeline_generator(agg_dicts)
            )),
            ('xgboost', XGBClassifier())])
    params['xgboost__n_estimators'] = list(np.arange(args.num_tree[0], args.num_tree[1], args.num_tree[2]))
    params['xgboost__max_depth'] = list(np.arange(args.depths[0], args.depths[1], args.depths[2]))
    params['xgboost__learning_rate'] = list(np.arange(args.lr[0], args.lr[1], args.lr[2]))
    params['xgboost__subsample'] = np.arange(args.subsample[0], args.subsample[1], args.subsample[2])
    params['xgboost__colsample_bytree'] = np.arange(args.colsample_bytree[0], args.colsample_bytree[1], args.colsample_bytree[2])
    try:
        grid_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=args.iter, cv=5, n_jobs=-1, scoring='f1')
        grid_search.fit(X, y)
    except:
        grid_search = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1, scoring='f1')
        grid_search.fit(X, y)
    best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
    return (best_parameters, score)

if __name__ == '__main__':
    result = main()
    print result




