import numpy as np
import scipy as sp
import pandas as pa
import math
import random as ra
from decisionTree import decisionTree
from sklearn import cross_validation
from Baggings import Baggings
from RandomForest import RandomForest
import sys


def main(data, F_T, k):
    result_list_T = []
    m, feature = data.shape
    names = list(range(feature))
    data.columns = names
    for F in F_T:
        result_list = []
        if k == 1:
            result = RandomForest(F, data.copy())
            testset = np.array(data)
            # for Entire dataset, the training and testing errors are equal
            train_error_rate = result.PredictionResult(testset)
            test_error_rate = train_error_rate
            result_list.append((str(k), str(F), train_error_rate, test_error_rate))
        else:
            cv = cross_validation.KFold(m, n_folds=k)
            for traincv, testcv in cv:
                trainset = data.loc[traincv, :].copy()
                testset = data.loc[testcv, :].copy()
                result = RandomForest(F, trainset)
                trainset1 = np.array(trainset)
                testset1 = np.array(testset)
                train_error_rate = result.PredictionResult(trainset1)
                test_error_rate = result.PredictionResult(testset1)
                result_list.append((str(k), str(F), train_error_rate, test_error_rate))
        result_list_T.append(result_list)
    return result_list_T


# print out the result

def display(result_list_T):
    for s in result_list_T:
        for i in range(len(s)):
            Fold, F, train_error_rate, test_error_rate = s[i]
            print(Fold + " Fold CV, " + str(i+1) + " Fold, " + F + " features," + " train_error: " + str(train_error_rate) + ", test_error: " + str(test_error_rate))


data = pa.read_table(sys.argv[1], sep=',', header=None)
data.drop([1],inplace=True,axis=1)
cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols]
k = int(sys.argv[3])
F_T = list(map(int, sys.argv[2].strip("[]").split(",")))

if __name__ == '__main__':
    result_list_T = main(data, F_T, k)
    display(result_list_T)

