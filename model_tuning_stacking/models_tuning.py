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
from subroutine import translate

parser = argparse.ArgumentParser(description='tuning various models')
parser.add_argument('--labels-file', type=str, required=True,
                    help='labels for supervised learning')
parser.add_argument('--nclass', type=str, required=True,
                    help='choose one out of 9 classes')
parser.add_argument('--data-dir', type=str, default='pooling_data',
                    help='the input data directory')
parser.add_argument('--dataset', type=str, required=True,
                    help='dataset name')
parser.add_argument('--dimension-reduction', type=str, default='pca',
                    choices=['pca', 'randomforest'],
                    help='dimension reduction methods')
parser.add_argument('--num', nargs='+', type=str, required=True,
                    help='dimension after reduction')
parser.add_argument('--num-tree', nargs='+', type=int, default=[100, 101, 1],
                    help='# of trees for tree based methods')
parser.add_argument('--depths', nargs='+', type=int, default=[3, 4, 1],
                    help='depths for tree based methods')
parser.add_argument('--lr', nargs='+', type=float, default=[1, 2, 1],
                    help='learning rate for gradient based model')
parser.add_argument('--alpha', nargs='+', type=float, default=[1, 2, 1],
                    help='rate for penalty')
parser.add_argument('--iter', type=int, default=20,
                    help='# of random sample')
args = parser.parse_args()

# load data and labels
data_path = os.path.join(args.data_dir, args.dataset)
X = pd.read_csv(data_path)
X.drop('business_id', axis=1, inplace=True)
y = pd.read_csv(args.labels_file)[args.nclass]
# potential methods
methods = {'pca': (PCA(), 'reduce_dim__n_components'),
            'randomforest': (SelectFromModel(RandomForestClassifier(n_estimators=200)), 'reduce_dim__threshold')}

# Handle random forest and pca differently
tmp = list(map(lambda x: int(x) if args.dimension_reduction == 'pca' else x, args.num))

# Dimension reduction and classifier pipline
# For classifier with too many parameters, we use randomizedsearchcv otherwise grid_search_cv
# forest pca/randomforest pipeline (random forest and extra tree)
def forest_method(X, y, est):
    estimators = [('reduce_dim', methods[args.dimension_reduction][0]), ('classifier', est)]
    clf = Pipeline(estimators)
    params = translate(methods, args.dimension_reduction, tmp)
    params['classifier__n_estimators'] = list(np.arange(args.num_tree[0], args.num_tree[1], args.num_tree[2]))
    params['classifier__max_depth'] = list(np.arange(args.depths[0], args.depths[1], args.depths[2]))
    params['classifier__max_features'] = ['auto', 'sqrt', 'log2']
    try:
        grid_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=args.iter, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
    except:
        grid_search = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
    best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
    return (best_parameters, score)


# xgboost pca/randomforest pipline
def xg_boost(X, y):
    estimators = [('reduce_dim', methods[args.dimension_reduction][0]), ('classifier', XGBClassifier())]
    clf = Pipeline(estimators)
    clf.set_params(classifier__objective="binary:logistic")
    params = translate(methods, args.dimension_reduction, tmp)
    params['classifier__n_estimators'] = list(np.arange(args.num_tree[0], args.num_tree[1], args.num_tree[2]))
    params['classifier__max_depth'] = list(np.arange(args.depths[0], args.depths[1], args.depths[2]))
    params['classifier__learning_rate'] = list(np.arange(args.lr[0], args.lr[1], args.lr[2]))
    params['classifier__subsample'] = [0.5, 0.8, 1]
    params['classifier__colsample_bytree'] = [0.5, 0.8, 1]
    try:
        grid_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=args.iter, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
    except:
        grid_search = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
    best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
    return (best_parameters, score)


# Adaboost pca/randomforest piplline
def adaboost(X, y):
    estimators = [('reduce_dim', methods[args.dimension_reduction][0]),
    ('classifier', AdaBoostClassifier())]
    clf = Pipeline(estimators)
    params = translate(methods, args.dimension_reduction, tmp)
    params['classifier__n_estimators'] = list(np.arange(args.num_tree[0], args.num_tree[1], args.num_tree[2]))
    params['classifier__learning_rate'] = list(np.arange(args.lr[0], args.lr[1], args.lr[2]))
    try:
        grid_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=args.iter, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
    except:
        grid_search = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
    best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
    return (best_parameters, score)


# SGD classifier
# SGD classifier pca/randomforest pipline(huber, logistic,svm,modified svm, etc)
def SGDclass(X, y, loss):
    estimators = [('reduce_dim', methods[args.dimension_reduction][0]), ('classifier', SGDClassifier())]
    clf = Pipeline(estimators)
    clf.set_params(classifier__loss=loss)
    params = translate(methods, args.dimension_reduction, tmp)
    params['classifier__alpha'] = list(np.arange(args.alpha[0], args.alpha[1], args.alpha[2]))
    try:
        grid_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=args.iter, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
    except:
        grid_search = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
    best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
    return (best_parameters, score)


# Discriminant analysis
# Discriminant analysis pca/randomforest pipeline (linear, quadratic)
def DiscAna(X, y, est):
    estimators = [('reduce_dim', methods[args.dimension_reduction][0]), ('classifier', est)]
    clf = Pipeline(estimators)
    params = translate(methods, args.dimension_reduction, tmp)
    if re.match(r'Linear*', str(est)) and args.dimension_reduction == 'pca':
        prop = np.append(np.arange(0.5, 1, 0.2), 1)
        outer = np.outer(prop, np.array(tmp))
        comp = outer.flatten().astype(int)
        params['classifier__n_components'] = list(comp)

    try:
        grid_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=args.iter, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
    except:
        grid_search = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1)
        grid_search.fit(X, y)
    best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
    return (best_parameters, score)


if __name__ == '__main__':
    def printer(d, key):
        print key + ':' + ' ' + str(d[key])
    d = {}
    d['xgboost'] = xg_boost(X, y)
    printer(d, 'xgboost')
    print 'xgBoosting Done!'
    d['random_forest'] = forest_method(X, y, RandomForestClassifier())
    printer(d, 'random_forest')
    print 'Random Forest Done!'
    d['extra_tree'] = forest_method(X, y, ExtraTreesClassifier())
    printer(d, 'extra_tree')
    print 'Extra Tree Done!'
    d['adaboost'] = adaboost(X, y)
    printer(d, 'adaboost')
    print 'Adaboost Done!'
    d['huber'] = SGDclass(X, y, 'modified_huber')
    printer(d, 'huber')
    print 'Huber loss Done!'
    d['logistic'] = SGDclass(X, y, 'log')
    printer(d, 'logistic')
    print 'Logtistic Done!'
    d['svm'] = SGDclass(X, y, 'hinge')
    printer(d, 'svm')
    print 'SVM Done!'
    d['msvm'] = SGDclass(X, y, 'squared_hinge')
    printer(d, 'msvm')
    print 'Modified SVM Done!'
    d['lda'] = DiscAna(X, y, LinearDiscriminantAnalysis())
    printer(d, 'lda')
    print 'LDA Done!'
    d['qda'] = DiscAna(X, y, QuadraticDiscriminantAnalysis())
    printer(d, 'qda')
    print 'QDA Done!'
    d = collections.OrderedDict(sorted(d.items(), key=lambda x: x[1][1], reverse=True))
    with open('Result_for_Class_' + args.nclass + '_' + args.dataset[:args.dataset.find('.')]+'.txt', 'w+') as f:
        for k, v in d.items():
            f.write(k)
            f.write('\t')
            f.write(str(v))
            f.write('\n')
    print 'All Done!'
