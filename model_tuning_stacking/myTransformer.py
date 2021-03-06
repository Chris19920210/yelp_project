import sys
import math
import numpy as np
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
import xgboost
import os
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
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
from sklearn.base import BaseEstimator, TransformerMixin
from subroutine import PartialStringColumns


class MyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        try:
            prediction = self.model.predict_proba(X)[:, 1]
        except:
            prediction = self.model.predict(X)
        return pd.DataFrame(prediction)


# partition data set by pooling method and diffferen network
class DataFrameSeparator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, **transform_params):
        tmp = X.columns
        pattern = r'(.*)_\d+_\d+$'
        d = set(map(lambda x: re.search(pattern, x).group(1), tmp))
        feature = {}
        for k in d:
            part = re.escape(k) + r'_\d+.*'
            feature[k] = X.filter(regex=part)
        return feature


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

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
