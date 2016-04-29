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
from subroutine import bestmodel, parse, bestinfo

# choose the model that give us the highest f1 score
# predict for the X_test
# read the labels from local
y = pd.read_csv('./business_id_labels.csv')

# potential methods
methods = {
    'svm': SGDClassifier(loss='hinge'),
    'msvm': SGDClassifier(loss='squared_hinge'),
    'huber': SGDClassifier(loss='modified_huber'),
    'logistic': SGDClassifier(loss='log'),
    'xgboost': XGBClassifier(),
    'adaboost': AdaBoostClassifier(),
    'extra_tree': ExtraTreesClassifier(),
    'random_forest': RandomForestClassifier(),
    'lda': LinearDiscriminantAnalysis(),
    'qda': QuadraticDiscriminantAnalysis()
}

# meta data for best classifier
path_dicts={
    '1':('max_pooling_result2/Result_for_Class_1_Inception_train224.txt', 'pooling_data/Inception_train224.csv', 'pooling_data/Inception_test224.csv' ),
    '2':('max_pooling_result2/Result_for_Class_2_Inception_train224.txt','pooling_data/Inception_train224.csv', 'pooling_data/Inception_test224.csv'),
    '3':('max_pooling_result2/Result_for_Class_3_Inception_train224.txt', 'pooling_data/Inception_train224.csv', 'pooling_data/Inception_test224.csv' ),
    '4':('max_pooling_result2/Result_for_Class_4_Inception_train224.txt', 'pooling_data/Inception_train224.csv', 'pooling_data/Inception_test224.csv' ),
    '5':('mean_pooling_result2/Result_for_Class_5_Inception_train224.txt', 'pooling_data2/Inception_train224.csv', 'pooling_data2/Inception_test224.csv' ),
    '6':('mean_pooling_result2/Result_for_Class_6_Inception_train224.txt','pooling_data2/Inception_train224.csv', 'pooling_data2/Inception_test224.csv' ),
    '7':('mean_pooling_result2/Result_for_Class_7_Inception_train224.txt', 'pooling_data2/Inception_train224.csv', 'pooling_data2/Inception_test224.csv' ),
    '8':('max_pooling_result2/Result_for_Class_8_Inception_train224.txt', 'pooling_data/Inception_train224.csv', 'pooling_data/Inception_test224.csv' ),
    '9':('max_pooling_result2/Result_for_Class_9_Inception_train224.txt', 'pooling_data2/Inception_train224.csv', 'pooling_data2/Inception_test224.csv' )
}


# store the prediction result for best classifier
def main():
    result = defaultdict()
    for key, value in path_dicts.items():
        dicts = bestinfo(value[0], methods)
        X_train = pd.read_csv(value[1])
        X_train.drop('business_id', axis=1, inplace=True)
        dicts['X_train'] = X_train
        dicts['y_train'] = y[key]
        X_test = pd.read_csv(value[2])
        X_test.drop('business_id', axis=1, inplace=True)
        dicts['X_test'] = X_test
        result['class'+key] = bestmodel(**dicts)
    return result

if __name__ == '__main__':
    result = main()
    result = pd.DataFrame(result)
    result.to_csv('./best_result.csv', index=False)









