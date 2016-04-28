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
from subroutine import bestmodel
from subroutine import parse
from subroutine import RetrievalDict
from subrouine import bestinfo

# read the labels from local
y = pd.read_csv('./business_id_labels')

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

# meta data for best classifier


# store the prediction result for best classifier
result = defaultdict()
for key, value in path_dicts.items():
    dicts = bestinfo(value[0], methods)
    dicts['X_train'] = pd.read_csv(value[1])
    dicts['y_train'] = y[str(key)]
    dicts['X_test']=pd.read_csv(value[2])
    result['class'+str(key)]=bestinfo(**dicts)








