import numpy as np
import scipy as sp
import math
import sys
import random

#sigmoid function
def sigmoid(x):
    f = []
    for x1 in x:
        sig = 1/(1+np.exp(-x1))
        f.append(sig)
    return f

# regularize the features in order to optimize the function conveniently 
def regular(x):
    f = []
    for i in range(1,len(x[0])-1):
        if np.std(x[:,i]) == 0:
            x1 = np.zeros(len(x))
        else:
            u = np.mean(x[:,i])
            v = np.std(x[:,i])
            x1 = (np.array(x[:,i])-u*np.ones(len(x)))/v
        f.append(x1)
    f = np.transpose(f)
    new = np.ones(len(x))
    f = np.insert(f,len(f[0]),new,axis=1)
    f = np.insert(f,0,x[:,0],axis=1)
    return f

#gradient descent method for optimization
def LogisticRegression(data):
    data = regular(data)
    alpha = 0.4
    data_tr = np.transpose(data)
    data_1 = np.delete(data_tr, 0 , axis=0)
    theta = np.zeros(len(data[0])-1)
    for i in range(250):
        linear = np.dot(data[:,1:],theta)
        estimate = sigmoid(linear)
        diff = np.array(data[:,0])-np.array(estimate)
        refresh = np.dot(data_1,diff)/len(data)
        theta += alpha*refresh
    return theta


def main():
    a = []
# read data
    with open(sys.argv[1],'r') as f:
        for line in f:
            a.append(list(map(float,line.strip().split(","))))
# add the column for constants
    new = np.ones(len(a))
    a = np.insert(a,len(a[0]),new,axis=1)
# divide the dataset by their response, which is used for testing and generate intended training set
    C0 = a[a[:,0]==0]
    C1 = a[a[:,0]==1]
    summary = []
    percentage = list(map(int,sys.argv[3].strip().split(",")))
    for j in percentage:
        error1 = []
        for i in range(100):
# generate train set and test set
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
            
            train = np.vstack((train0,train1))
            test = np.vstack((test0,test1))
# train the model
            w = LogisticRegression(train)
# count the error
            error = 0
            for element in regular(test):
                w1 = np.dot(element[1:],w)
                q = 1/(1+np.exp(-w1))
                if q > 0.5:
                    q = 1
                else:
                   q = 0
                if q!= element[0]:
                    error += 1
    
            error1.append(error/len(test))
            
        summary.append(error1)        
# print out the result
    summary = np.array(summary)
    for i in range(len(percentage)):
        mean = np.mean(summary[i])
        std = np.std(summary[i])
        print(str(percentage[i])+" "+str(mean)+" "+str(std))

main()

    

        
    
