#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split


class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        num2label = dict((i, label)for label, i in self.label2num.items())
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self


def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)



def main():
    X = np.load('/home/chris/Microsoft_malware/X_train.npy')
    Y = np.load('/home/chris/Microsoft_malware/label_y.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    clf = XGBoostClassifier(
        eval_metric='auc',
        num_class=9,
        nthread=2,
        eta=0.1,
        num_boost_round=200,
        max_depth=12,
        subsample=1,
        colsample_bytree=1.0,
        silent=1,
        )

    params = {
        'clf__num_boost_round': [100, 150, 200],
        'clf__max_depth': [3, 6, 8],
    }

    # Cross_validation for grid search
    grid_search = GridSearchCV(clf, params, n_jobs=1, cv=5)
    grid_search.fit(X_train, y_train)
    best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    result = accuracy_score(y_test, grid_search.predict(X_test))
    print("Predict Accuracy: " + str(result))
    print("XGboost using raw pixel features:\n%s\n" % (metrics.classification_report(y_test, grid_search.predict(X_test))))



if __name__ == '__main__':
    main()

