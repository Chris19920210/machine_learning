import os.path
import time
import numpy as np
from numpy import matrix as mat
from sklearn import preprocessing as pre
import random
from numpy.linalg import norm
import sys
import math


#get the value for objective function
def prim(w, lambda1, dataX, Y, m):
    sum1 = (lambda1/2)*(norm(w)**2)
    part2 = np.ones(m)-np.array(Y.T)*np.array((dataX*w).T)
    part2 = np.array([max([0, i]) for i in part2[0]])
    sum1 += (1/m)*np.sum(part2)
    return sum1


def mysgdsvm():
    data = []
    
    with open(sys.argv[1], 'r') as f:
        for line in f:
            data.append(list(map(float, line.strip().split(','))))
    #data prepossessing
    data = np.array(data)
    data = data[:, (data != 0).sum(axis=0) > 0]
    # standardized the data
    dataX = mat(pre.scale(data[:, 1:]))
    # change the response to the binary data -1 and 1
    Y = mat([-x+2 for x in data[:, 0]]).T
    # split the data according to their responses
    # and get the shape for each subset
    dataX_1 = dataX[np.array(Y.T == 1)[0], :]
    Y_1 = Y[np.array(Y.T == 1)[0], :]
    y1_m, y1_n = Y_1.shape
    dataX_m1 = dataX[np.array(Y.T == -1)[0], :]
    Y_m1 = Y[np.array(Y.T == -1)[0], :]
    ym1_m, ym1_n = Y_m1.shape
    m, n = dataX.shape
    # set the initial values and constant for penalty term
    w = mat(np.zeros((n, 1)))
    T = 1
    lambda1 = 10
    primalmaximum = []
    while (T < 10000):
        w_old = w.copy()
        # random sample
        # to prevent that the half of minibatch size greater
        # than any of the subset
        if 0.5*float(sys.argv[2]) >= min([ym1_m, y1_m]):
            k = int(sys.argv[2])
            eta = 1/(lambda1*T)
            Y_train = Y
            dataX_train = dataX
        else:
            # get the value for eta
            eta = 1/(lambda1*T)
            k = int(sys.argv[2])
            # randomly choose the indexes for training set 
            A1_index = random.sample(range(0, y1_m), math.ceil(k/2))
            Am1_index = random.sample(range(0, ym1_m), math.floor(k/2))
            # get the train set
            data_train1 = dataX_1[A1_index, :]
            Y_train1 = Y_1[A1_index]
            data_trainm1 = dataX_m1[Am1_index, :]
            Y_trainm1 = Y_m1[Am1_index]
            Y_train = np.concatenate((Y_train1, Y_trainm1))
            dataX_train = np.concatenate((data_train1, data_trainm1))

        # choose A+ set that need to be used to refresh the gradient
        A_index_p = (np.array(Y_train.T)*np.array((dataX_train*w).T) < np.ones(k))[0]
        dataX_train_p = dataX_train[A_index_p, :]
        Y_train_p = Y_train[A_index_p]
        # refresh w
        w = (1-eta*lambda1)*w + (eta/k)*(Y_train_p.T*dataX_train_p).T
        # project back
        w = min([1, (1/lambda1**0.5)/(norm(w))])*w
        # calculate the value for objective function
        tmp = prim(w, lambda1, dataX, Y, m)
        primalmaximum.append(tmp)
        T += 1
        # calculate the differences between w_old and w
        dist = np.linalg.norm(w_old-w)
        if dist <= 0.001:
            break
    return primalmaximum

times = []
save_path = './'
for i in range(int(sys.argv[3])):
    start_time = time.time()
    result = mysgdsvm()
    times.append(time.time()-start_time)
    result = np.array(result)
    np.savetxt(save_path+"Rihan-" + sys.argv[2] + "-SGD-"+str(i)+".txt", result)


times = np.array(times)
print("Avg runtime: " + "with minibatch size = " + sys.argv[2] + " " + "is " + str(np.average(times)) + "s")
print("Std runtime: " + "with minibatch size = " + sys.argv[2] + " " + "is " + str(np.std(times)) + "s")
print("Plot data have been plolted in ./Rihan-" + sys.argv[2] + "-SGD-k.txt")










