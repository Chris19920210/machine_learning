import numpy as np
import scipy as sp
import pandas as pa
import math
import random as ra


class decisionTree():
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k
        self.labels = np.unique(self.y)
        self.dataset = self.dataTrans()
        self.split_list = []
        self.splitTree(self.dataset)

    def dataTrans(self):
        data = pa.concat([self.y, self.X], axis=1)
        data = data.reset_index(drop=True)
        return data

    def Entropy(self, Y):
        # length of Y
        Y = np.array(Y)
        n = len(Y)
        if n == 0:
            return float("inf")
        # store the proportion for different class
        a = []
        for s in self.labels:
            tmp = len(Y[Y == s])
            a.append(tmp)
        # get the proportion
        a = np.array(a)/n
        # log value
        log_a = np.log2(a)
        # get the entropy
        ent = -sum(a*log_a)
        return ent

    def select_elements(self, seq, perc=50):
        # Select a defined percentage of the elements of seq.
        return seq[::int(100.0/perc)]

    # Calculate the value to cut a given feature
    def FeatrueSplit(self, subdata, i):
        # sorted by x
        subdata = subdata.sort_values([i])
        data_array = np.array(subdata)
        # transfrom to array
        X_list = np.array(subdata.loc[:, [i]]).T
        X_list = X_list[0]
        n = len(X_list)
        # get the middle value
        X_list_F = np.unique((X_list[1:] + X_list[:-1])/2)
        # Select k percentage,default is 50 percentage
        X_list_Entro = self.select_elements(X_list_F)
        z = 0
        final_condition = float("inf")
        for j in range(len(X_list_Entro)):
            T1 = data_array[data_array[:, 1] <= X_list_Entro[j]][:, 0]
            T2 = data_array[data_array[:, 1] > X_list_Entro[j]][:, 0]
            y1 = self.Entropy(T1)
            y2 = self.Entropy(T2)
            conditional_entropy = (len(T1)/n)*y1+(len(T2)/n)*y2
            if conditional_entropy < final_condition:
                final_condition = conditional_entropy
                z = j
        if z == 0:
            f_data = subdata[subdata.loc[:, i] <= X_list_Entro[z]]
            f_index_list = f_data.index.values
            f_index = ra.choice(f_index_list)
        else:
            f_data = subdata[(subdata.loc[:, i]>=X_list_Entro[z-1]) & (subdata.loc[:, i] < X_list_Entro[z])]
            f_index_list = f_data.index.values
            f_index = ra.choice(f_index_list)

        Split_value = subdata.loc[f_index,i]

        return (Split_value, final_condition)

    def majorityVote(self, Y):
        Y = np.array(Y)
        n = len(Y)
        if n == 0:
            return ra.choice(self.labels)
        # store the proportion for different class
        a = []
        for s in self.labels:
            tmp = len(Y[Y == s])
            a.append(tmp)
        a = np.array(a)
        index = a.argmax()
        label = self.labels[index]
        return label

    def leftChild(self, i):
        return 2*i+1

    def rightChild(self, i):
        return 2*i+2

    # return the best feature and its corresponding value for splitting
    def splitTree(self, dataset, k=1):
        if k <= self.k:
            # entropy
            final_conditional = float("inf")
            # bestsplit_featrue
            bestsplit_featrue = 0
            # split value
            final_split = 0
            n, feature = dataset.shape
            for i in dataset.columns.values[1:]:
                subdata = dataset.loc[:, [0, i]].copy()
                Split_value, cond_entropy = self.FeatrueSplit(subdata, i)
                if final_conditional > cond_entropy:
                    final_conditional = cond_entropy
                    final_split = Split_value
                    bestsplit_featrue = i

            dataset1 = dataset[dataset.loc[:, bestsplit_featrue] <= final_split].copy()
            dataset2 = dataset[dataset.loc[:, bestsplit_featrue] > final_split].copy()
            label1 = self.majorityVote(dataset1.loc[:, 0])
            label2 = self.majorityVote(dataset2.loc[:, 0])
            self.split_list.append((k, bestsplit_featrue, final_split, label1, label2))
            k = k + 1
            # building the tree recursively
            self.splitTree(dataset1, k)
            self.splitTree(dataset2, k)
            self.split_list = sorted(self.split_list,key=lambda x:x[0])

    # test a given sample following the decision tree. Correct prediction is 0, wrong is 1
    def testThrough(self, data):
        total_depth = len(self.split_list)
        i = 0
        while(i < total_depth):
            z = self.split_list[i]
            layer, feature, value, label1, label2 = z
            if data[feature] <= value:
                label = label1
                i = self.leftChild(i)
            else:
                label = label2
                i = self.rightChild(i)
        if int(label) != int(data[0]):
            return 1
        else:
            return 0

    # for new data set, make a prediction and calculate the error rate
    def testError(self, newX, newY):
        X = np.mat(newX)
        Y = np.mat(newY)
        self.split_list = sorted(self.split_list, key=lambda x:x[0])
        data_test = np.concatenate((Y.T, newX), axis=1)
        data_test = np.array(data_test)
        n, features = data_test.shape
        count = 0
        for i in range(n):
            test = data_test[i, :]
            count += self.testThrough(test)
        error_rate = count/n
        return error_rate






