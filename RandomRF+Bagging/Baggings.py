import numpy as np
import scipy as sp
import pandas as pa
import math
import random as ra
from decisionTree import decisionTree
from sklearn import cross_validation

class Baggings():
    def __init__(self, B, data):
        self.B = B
        self.data = data
        self.trees = self.trainingBagging(self.B, self.data)

    def trainingBagging(self, B, data):
        # Train Bagging with size B
        m, feature = data.shape
        result = []
        for i in range(B):
            # bootstrap sampling for building a tree
            bootstrap_sample = data.sample(m, replace=True).copy()
            X1 = bootstrap_sample.loc[:, 1:]
            Y1 = bootstrap_sample.loc[:, 0]
            tmp = decisionTree(X1, Y1, 2)
            # get the result for each tree
            result.append(tmp)
        return result

    def PredictionResult(self, newData):
        m, feature = newData.shape
        count = 0
        for i in newData:
            count_ind = 0
            for s in self.trees:
                # get the prediction result from each tree
                count_ind += s.testThrough(i)
            # make the final prediction by majority vote
            if count_ind > 0.5*self.B:
                count += 1
        return count/m


