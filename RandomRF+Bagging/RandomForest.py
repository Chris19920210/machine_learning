import numpy as np
import scipy as sp
import pandas as pa
import math
import random as ra
from decisionTree import decisionTree
from Baggings import Baggings
from sklearn import cross_validation


# RandomForest inherit from baggings for the function PredictionResult
class RandomForest(Baggings):
    def __init__(self, F, data):
        self.F = F
        self.data = data
        self.trees = self.trainingRandom(self.F, self.data)
        self.B = 100

    def trainingRandom(self, F, data):
        # Train Random with # of feature = F
        m, feature = data.shape
        result = []
        for i in range(100):
            bootstrap_sample = data.sample(m, replace=True).copy()
            # Randomly choose given number of features
            X1 = bootstrap_sample.loc[:, np.random.choice(list(range(1, feature)), F, replace=False)]
            Y1 = bootstrap_sample.loc[:, 0]
            tmp = decisionTree(X1, Y1, 2)
            result.append(tmp)
        return result
    
    
    def PredictionResult(self, newData):
        return super().PredictionResult(newData)
    








