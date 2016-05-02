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
import pickle
# choose the model that give us the highest f1 score
# predict for the X_test
# read the labels from local
parser = argparse.ArgumentParser(description='best model predict various models')
parser.add_argument('--save-dir', type=str, default='.',
                    help='the input data directory')
args = parser.parse_args()

y = pd.read_csv('./business_id_labels.csv')
Id= pd.read_csv('./ID.csv')
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
    '1':('result2/Result_for_Class_1_all_data.txt', './all_data_train.csv', './all_data_test.csv' ),
    '2':('result2/Result_for_Class_2_all_data.txt','./all_data_train.csv', './all_data_test.csv'),
    '3':('result2/Result_for_Class_3_all_data.txt', './all_data_train.csv', './all_data_test.csv' ),
    '4':('result2/Result_for_Class_4_all_data.txt', './all_data_train.csv', './all_data_test.csv' ),
    '5':('result2/Result_for_Class_5_all_data.txt', './all_data_train.csv', './all_data_test.csv' ),
    '6':('result2/Result_for_Class_6_all_data.txt','./all_data_train.csv', './all_data_test.csv' ),
    '7':('result2/Result_for_Class_7_all_data.txt', './all_data_train.csv', './all_data_test.csv' ),
    '8':('result2/Result_for_Class_8_all_data.txt', './all_data_train.csv', './all_data_test.csv' ),
    '9':('result2/Result_for_Class_9_all_data.txt', './all_data_train.csv', './all_data_test.csv' )
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
        clf, result['class'+key] = bestmodel(**dicts)
        pickle.dump(clf, open(os.path.join(args.save_dir, 'class_'+ key +'.p'), "wb"))
    return result

if __name__ == '__main__':
    result = main()
    result = pd.DataFrame(result)
    result = pd.concat([Id, result], axis=1)
    result.to_csv(os.path.join(args.save_dir,'best_result2.csv'), index=False)









