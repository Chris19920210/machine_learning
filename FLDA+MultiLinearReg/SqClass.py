import numpy as np
import scipy as sp
import math
import sys
import random

def SqClass():
    a = []
    with open(sys.argv[1],'r') as f:
        for line in f:
            a.append(list(map(float,line.strip().split(","))))
    a_t = np.transpose(a)
    a_t =a_t[~(a_t==0).all(1)]
    a_u = np.transpose(a_t)
    new = np.ones(len(a_u))
    a_u = np.insert(a_u,len(a_u[0]),new,axis=1)
    test_error = []
    train_error = []
    T = []
    for i  in a_u[:,0]:
        if i == 1:
            row = np.array([1,0,0,0])
        elif i == 3:
            row = np.array([0,1,0,0])
        elif i == 7:
            row = np.array([0,0,1,0])
        elif i == 8:
            row = np.array([0,0,0,1])
        T.append(row)
    T = np.array(T)

    for i in range(int(sys.argv[2])):
        test = math.ceil(len(a_u)/int(sys.argv[2]))
        inde = random.sample(range(len(a_u)),test)
        train = np.delete(a_u, inde, axis=0)
        T_tr = np.delete(T, inde, axis=0)
        test = a_u[inde]
        T_te = T[inde]
        
        from numpy.linalg import inv            
        x_tx = np.dot(train[:,1:].T,train[:,1:]) + np.eye(len(train[0][1:]),len(train[0][1:]))
        x_tx_inv = inv(x_tx)
        w = np.dot(np.dot(np.matrix(x_tx_inv),train[:,1:].T),np.array(T_tr))

        Y_train = np.dot(train[:,1:],w)
        Y_test = np.dot(test[:,1:],w)                                                                      

        error = 0
        for i in range(len(train)):
            index1 = np.argmax(T_tr[i])
            index2 = np.argmax(Y_train[i])
            if index1 != index2:
                error += 1
            
        error1 = 0
        for i in range(len(test)):
            index1 = np.argmax(T_te[i])
            index2 = np.argmax(Y_test[i])
            if index1 != index2:
                error1 += 1

        train_error.append(error/len(train))
        test_error.append(error1/len(test))

    test_sd = np.std(test_error)
    train_sd = np.std(train_error)
    print("train_error: " + str(train_error))
    print("test_error: " + str(test_error))
    print("test_sd: " + str(test_sd))
    print("train_sd: " + str(train_sd))

SqClass()
