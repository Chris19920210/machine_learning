#!/usr/bin/python3
import numpy as np
import scipy as sp
import math
import random
import sys

class inner():
    def __init__(self,matrix):
        self.matrix = matrix
        
    def group_center(self):
        a = self.matrix[:,1:].sum(axis=0)/len(self.matrix)
        return a

    def inner_var(self):
        a = self.matrix[:,1:].sum(axis=0)/len(self.matrix)
        sum1 = np.zeros((len(self.matrix[1])-1,len(self.matrix[1])-1))
        
        for i in range(len(self.matrix)):
            s = np.array(self.matrix[i,1:])-np.array(a)
            instance = np.outer(s,s)
            sum1 += instance
        return sum1
    
class store():
    def __init__(self,center,inner_var):
        self.center = center
        self.inner_var = inner_var
        
class gaussian():
    def __init__(self,matrix):
        self.matrix = matrix
        
    def group_mean(self):
        a = np.array(self.matrix).sum(axis=0)/len(self.matrix)
        return a

    def inner_var(self):
        a = np.array(self.matrix).sum(axis=0)/len(self.matrix)
        sum1 = np.zeros((len(self.matrix[1]),len(self.matrix[1])))
        
        for i in range(len(self.matrix)):
            s = np.array(self.matrix[i])-np.array(a)
            instance = np.array(np.outer(s,s))
            sum1 += (1/len(self.matrix))*instance
        return sum1
    
class gaussianstore():
    def __init__(self,center,inner_var):
        self.center = center
        self.inner_var = inner_var
        
#Data preparation
def Fisher():
    a = []
    with open(sys.argv[1],'r') as f:
        for line in f:
            a.append(list(map(float,line.strip().split(","))))
    a_t = np.transpose(a)
    a_t =a_t[~(a_t==0).all(1)]
    a_u = np.transpose(a_t)
    test_error = []
    train_error = []
    for i in range(int(sys.argv[2])):
        test = math.ceil(len(a_u)/int(sys.argv[2]))
        inde = random.sample(range(len(a_u)),test)
        train = np.delete(a_u, inde, axis=0)
        test = a_u[inde]


    ## 3d-array for convenience
        name = [1,3,7,8]
        C=[]
        for i in range(4):
            C.append(train[train[:,0]==name[i]])
        D=[]
        for i in range(4):
            D.append(test[test[:,0]==name[i]])
    # Sw and Sb matrix
        var = []
        q = []

    ##class lists
        for i in range(4):
            q.append(inner(C[i]))

    ## store the inner variance and center for each group
        for i in range(4):
           var.append(store(q[i].group_center(),q[i].inner_var()))

    ## Sb and Sw:
        Sw = np.zeros((len(var[0].center),len(var[0].center)))
        for i in range(4):
            Sw += var[i].inner_var
    # invertible 
        Sw = Sw + np.eye(len(var[0].center),len(var[0].center))

        total_mean = train[:,1:].sum(axis=0)/len(train)

        Sb = np.zeros((len(var[0].center),len(var[0].center)))
        for i in range(4):
            s = (total_mean-var[i].center)
            Sb += len(C[i])*np.array(np.outer(s,s))

    # get w
        from numpy.linalg import inv
        Pr = np.matrix(inv(Sw))*np.matrix(Sb)
        from scipy.sparse.linalg import eigs
        value, vector = eigs(Pr,3)
    # as we have 4 class, hence we can project x into 3-dimension subspace
        w = np.transpose(vector)[0:3]
    ##projected space
        R=[]
        for s in range(len(C)):
            r1=[]
            for i in range(len(C[s])):
                w1 = np.dot(w,C[s][i][1:])
                r1.append(w1)
            R.append(np.array(r1))

        R1=[]
        for s in range(len(D)):
            r1=[]
            for i in range(len(D[s])):
                w1 = np.dot(w,D[s][i][1:])
                r1.append(w1)
            R1.append(np.array(r1))
        

    ##generative gaussian model
        Q = []
        GaussianInfo=[]
        for i in range(4):
            Q.append(gaussian(R[i]))

        for i in range(4):
            GaussianInfo.append(gaussianstore(Q[i].group_mean(),Q[i].inner_var()))

    # compare
        def greatest_posterior(x):
            q = []
            for i in range(len(GaussianInfo)):
                q.append(1/((math.pow(2*math.pi,3/2)*math.pow(np.linalg.det(GaussianInfo[i].inner_var),1/2)))*math.exp(-1/2*np.dot(np.dot((x-GaussianInfo[i].center),inv(GaussianInfo[i].inner_var)),(x-GaussianInfo[i].center)))*len(R[i])/len(train))
            index = q.index(max(q))
            return index
        def greatest_posterior1(x):
            q = []
            for i in range(len(GaussianInfo)):
                q.append(1/((math.pow(2*math.pi,3/2)*math.pow(np.linalg.det(GaussianInfo[i].inner_var),1/2)))*math.exp(-1/2*np.dot(np.dot((x-GaussianInfo[i].center),inv(GaussianInfo[i].inner_var)),(x-GaussianInfo[i].center)))*len(R1[i])/len(test))
            index = q.index(max(q))
            return index

        error = 0

        for s in range(4):
            for i in range(len(R[s])):
                q = greatest_posterior(R[s][i])
                if q != s:
                    error += 1

        error1 = 0
        for s in range(4):
            for i in range(len(R1[s])):
                q = greatest_posterior1(R1[s][i])
                if q != s:
                    error1 += 1

        train_error.append(error/len(train))
        test_error.append(error1/len(test))
        

    test_sd = np.std(test_error)
    train_sd = np.std(train_error)
    print("train_error: " + str(train_error))
    print("test_error: " + str(test_error))
    print("test_sd: " + str(test_sd))
    print("train_sd: " + str(train_sd))
    
Fisher()






        


