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
import ast
from collections import defaultdict
import itertools
import pickle

def translate(methods, k, tmp):
    params = {}
    key = methods[k][1]
    params[key] = tmp
    return params


def path_generator(path_tuple):
    result = []
    for value in path_tuple:
        data_dir, dataset = value
        path = os.path.join(data_dir, dataset)
        result.append(path)
    return result


def parse(line):
    d = {}
    partition = re.search(r'(\w+).*({.*}).*(\d+\.\d+)', line)
    model = partition.group(1)
    params = partition.group(2)
    score = partition.group(3)
    params = ast.literal_eval(params)
    pattern = r'reduce*'
    # filter and parse the parameter
    filter_dict_reduce = {k[k.find('__'):][2:]: v for k, v in params.items() if re.match(pattern, k)}
    filter_dict_classifier = {k[k.find('__'):][2:]: v for k, v in params.items() if not re.match(pattern, k)}
    d['reduce'] = filter_dict_reduce
    d['classifier'] = filter_dict_classifier
    return (model, d, float(score))


def PartialStringColumns(*args):
    tmp = list(itertools.product(args[0], args[1]))
    d = {v[0] + '_' + v[1]: v for v in tmp}
    return d


def RetrievalDict(key, path, threshold, dicts):
    d = defaultdict(dict)
    with open(path, 'r') as f:
        for k, line in enumerate(f):
            if k < threshold:
                line = line.strip()
                model, d_individual, _ = parse(line)
                classifier = dicts[model]
                d[key][str(k)] = (d_individual, classifier)
    return d


# generate a list for FeatureUnion
# nested dictionary
def pipeline_generator(dicts):
    result = []
    for key, value in dicts.items():
        for k, v in value.items():
            d_individual, classifier = v
            chunk = (str(key)+ '_'+ str(k), Pipeline([
                    ('selector', DataFrameSelector(key=key)),
                    ('reduce_dim', PCA(**d_individual['reduce'])),
                    ('classifier', MyTransformer(classifier.set_params(**d_individual['classifier'])))
                ]))
            result.append(chunk)
    return result


# best model parser and produce the prediction result for test data
def bestmodel(path,name,**kwargs):
    estimator = [('pca', PCA(**kwargs['meta']['reduce'])), ('classifier', kwargs['classifier'].set_params(**kwargs['meta']['classifier']))]
    clf = Pipeline(estimator)
    clf.fit(kwargs['X_train'], kwargs['y_train'])
    return (clf, clf.predict(kwargs['X_test']))


# generate the dictionary for tracing the best classifier for each case(the first line of the result)
def bestinfo(path_to_model, methods):
    result = defaultdict()
    with open(path_to_model, 'r') as f:
        for key, line in enumerate(f):
            if key < 1:
                line = line.strip()
                model, d_individual, _ = parse(line)
                classifier = methods[model]
                result['classifier'] = classifier
                result['meta'] = d_individual
                return result
















