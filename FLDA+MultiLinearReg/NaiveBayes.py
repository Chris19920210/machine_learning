import numpy as np
import scipy as sp
import sys
import random
import math
from numpy.linalg import inv

# As the naivee bayes is actually train each class seperately, hence we use the oop in order to conveniently train each class
class gaussian():
    def __init__(self,matrix):
        self.matrix = matrix
        # calculate the mean for each class
    def group_mean(self):
        a = np.array(self.matrix).sum(axis=0)/len(self.matrix)
        return a
# calculate the variance-corvariance  for each class
    def inner_var(self):
        var1 = np.var(self.matrix,axis=0)
        return var1
   # store the mean and corvariance for each class in order to predict conveniently
class gaussianstore():
    def __init__(self,center,inner_var):
        self.center = center
        self.inner_var = inner_var
    # gausssian kernel function for calculate the posterior probability
def kernel(x,mean,var):
    q = 1
    for i in range(len(mean)):
        q = q*(1/math.pow(2*math.pi*var[i],1/2))*math.exp(-1/2*math.pow(x[i]-mean[i],2)/var[i])
    return q
    
def main():
# read data
    a = []
    with open(sys.argv[1]) as f:
        for line in f:
            a.append(list(map(float,line.strip().split(","))))
    a = np.array(a)
# split the data by their response in order to train the model conveniently
    C0 = a[a[:,0]==0]
    C1 = a[a[:,0]==1]
    percentage = list(map(int,sys.argv[3].strip().split(",")))
    summary = []
    for j in percentage:
        error1 = []
        for i in range(100):
# split the training set and test set by requirement
            train0_volume = math.ceil(float(sys.argv[2])*len(C0))
            train0_index = random.sample(range(len(C0)),train0_volume)
            test0 = np.delete(C0,train0_index,axis = 0)
            train0_tmp = C0[train0_index]
            index_for_train = random.sample(range(len(train0_index)),math.ceil(train0_volume*j/100))
            train0 = train0_tmp[index_for_train]
            
            train1_volume = math.ceil(float(sys.argv[2])*len(C1))
            train1_index = random.sample(range(len(C1)),train1_volume)
            test1 = np.delete(C1,train1_index,axis = 0)
            train1_tmp = C1[train1_index]
            index_for_train = random.sample(range(len(train1_index)),math.ceil(train1_volume*j/100))
            train1 = train1_tmp[index_for_train]

            test = np.vstack((test0,test1))
# train the model and store their mean and variance matrix
            q = []
            model_info = []
            q.append(gaussian(train0[:,1:]))
            q.append(gaussian(train1[:,1:]))
            model_info.append(gaussianstore(q[0].group_mean(),q[0].inner_var()))
            model_info.append(gaussianstore(q[1].group_mean(),q[1].inner_var()))

# remove the the feature which has variance equal to 0(which means that for the training the data the particular feature is all zero)
            index0 = np.where(model_info[0].inner_var == 0)[0]
            if len(index0) != 0:
                model_info[0].center = np.delete(model_info[0].center,index0)
                model_info[0].inner_var = np.delete(model_info[0].inner_var,index0)
                index0 +=1
                test_0 = np.delete(test,index0,axis=1)
            else:
                test_0 = test
                
                

            index1 = np.where(model_info[1].inner_var == 0)[0]
            if len(index1) != 0:
                model_info[1].center = np.delete(model_info[1].center,index1)
                model_info[1].inner_var = np.delete(model_info[1].inner_var,index1)
                index1 +=1
                test_1 = np.delete(test,index1,axis=1)
            else:
                test_1 = test


  # calculate the posterior probability          
            Q0 = []
            for element0 in test_0:
                q0 = kernel(element0[1:],model_info[0].center,model_info[0].inner_var)*(len(train0)/(len(train0)+len(train1)))
                Q0.append(q0)
            Q1 = []
            for element1 in test_1:
                q1 = kernel(element1[1:],model_info[1].center,model_info[1].inner_var)*(len(train1)/(len(train0)+len(train1)))
                Q1.append(q1)
 # count the errors 
            error = 0
            for i in range(len(test)):
                if Q0[i] > Q1[i]:
                    q = 0
                else:
                    q = 1
                if q != test[i,0]:
                    error += 1
                 
            error1.append(error/len(test))
            
        summary.append(error1)        
    summary = np.array(summary)
   # print out the result
    for i in range(len(percentage)):
        mean = np.mean(summary[i])
        std = np.std(summary[i])
        print(str(percentage[i])+" "+str(mean)+" "+str(std))

main()

                
