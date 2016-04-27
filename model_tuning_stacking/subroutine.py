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
import ast
from collections import defaultdict

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


def PartialStringColumns(columns):
    d = defaultdict(str)
    for value in columns:
        partition = re.search(r'^([a-zA-Z]+)_(.*)(_\d+_\d+)', value)
        method = partition.group(1)
        data = partition.group(2)
        d[method + '_' + data] = (method, data)
    return d

def RerivalDict(key, path, lists, threshold):

    with open(path, 'r') as f:
        for k, line in enumerate(f):
            if k < threshold
            line = line.strip()
            model, d_individual, _ = parse(line)



def pipeline_generator(dicts):







