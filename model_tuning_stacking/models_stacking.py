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
from subroutine import parse,  RetrivalDict
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
args = parser.parse_args()



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

key, path, lists, threshold, dicts
#  data preparation
#  generate metadata dictionary
dict_list = []
# path to mean_pooling_result
path1 = 'mean_pooling_result'
# path to mean_pooling_result
path2 = 'max_pooling_result'

# files that with specific labels
files = [f for f in os.listdir(path1) if re.match(r'Result_for_Class_' + re.escape(args.nclass) + r'.*', f)]

# pattern in order to retrive file by args
pattern = r"^(Result_for_Class_)\d+_(\w+.*)_\w+\d+\.txt$"


for value in args.mean_pooling + args.max_pooling:
    if re.match(r'mean.*', value):
        # retrive wanted file
        tmp_list = [f for f in files if re.search(pattern, f).group(2) == value[value.find('_'):][1:]]
        # path to wanted file
        path = os.path.join(path1, tmp_list[0])
        d = RetrivalDict(value, path, args.top, methods)
    else:
        tmp_list = [f for f in files if re.search(pattern, f).group(2) == value[value.find('_'):][1:]]
        path = os.path.join(path1, tmp_list[0])
        d = RetrivalDict(value, path, args.top, methods)
    dict_list.append(d)

# transform to dictionary
agg_dicts = defaultdict(d)
for value in dict_list:
    k, v = value.items()
    agg_dicts[k] = v







pipeline = Pipeline([
    # Extract the subject & body
    ('subjectbody', SubjectBodyExtractor()),

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the post's subject line
            ('subject', Pipeline([
                ('selector', ItemSelector(key='subject')),
                ('tfidf', TfidfVectorizer(min_df=50)),
            ])),

            # Pipeline for standard bag-of-words model for body
            ('body_bow', Pipeline([
                ('selector', ItemSelector(key='body')),
                ('tfidf', TfidfVectorizer()),
                ('best', TruncatedSVD(n_components=50)),
            ])),

            # Pipeline for pulling ad hoc features from post's body
            ('body_stats', Pipeline([
                ('selector', ItemSelector(key='body')),
                ('stats', TextStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

        ],

        # weight components in FeatureUnion
        transformer_weights={
            'subject': 0.8,
            'body_bow': 0.5,
            'body_stats': 1.0,
        },
    )),

    # Use a SVC classifier on the combined features
    ('svc', SVC(kernel='linear')),
])

# general pipeline
pipeline = Pipeline([
        ('separator', DataFrameSeparator()),
        ('union', FeatureUnion(
         transformer_list=[

         ]
         )),
        ('xgboost', XGBClassifier())]
