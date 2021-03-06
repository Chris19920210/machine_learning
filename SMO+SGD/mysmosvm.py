import os.path
import time
import numpy as np
from numpy import matrix as mat
from sklearn import preprocessing as pre
from random import randint
import sys


#evaluate the f(Xi)
def evaluate(i, alphas, Y, dataX, b):
    aY = np.array(alphas.T)*np.array(Y.T)
    sum1 = np.dot(aY, dataX*dataX[i].T)+b
    return sum1

# random select alpha_j which is not equal to i
def randomselect(i, m):
    x = i
    while(x == i):
        x = randint(0, m-1)
    return x

# get clipped alpha_j after each iteration
def clipalpha(alphaJ, L, H):
    if alphaJ > H:
        alphaJ = H
    elif alphaJ < L:
        alphaJ = L
    return alphaJ

# get the value for objective function
def dualObjective(alphas, Y, dataX):
    alphas = np.array(alphas)
    Y = np.array(Y)
    aY = mat(alphas*Y)
    sum1 = sum(alphas)
    sum1 += float(-0.5*aY.T*dataX*dataX.T*aY)
    return sum1


def smoSimple():
    data = []
    # read the data
    with open(sys.argv[1], 'r') as f:
        for line in f:
            data.append(list(map(float, line.strip().split(','))))
    # data prepossessing
    data = np.array(data)
    data = data[:, (data != 0).sum(axis=0) > 0]
    dataX = mat(pre.scale(data[:, 1:]))
    # change response to binary -1,1
    Y = mat([-x+2 for x in data[:, 0]]).T
    m, n = dataX.shape
    # set  initial value and constant for penalty term and numerical tolerance
    alphas = mat(np.zeros((m, 1)))
    C = 10
    b = 0
    tolerance = 0.0001
    dualmaximum = []
    passes = 0
    # maximum value for iteration
    while (passes < 10000):

        alphas_old = alphas.copy()

        for i in range(m):
            # Evaluate the model i
            fXi = evaluate(i, alphas, Y, dataX, b)
            Ei = fXi - float(Y[i])

            # Check if we can optimize (alphas always between 0 and C)
            if ((Y[i] * Ei < -tolerance) and (alphas[i] < C)) or \
               ((Y[i] * Ei > tolerance) and (alphas[i] > 0)):

                # Select a random J which is not equal to
                j = randomselect(i, m)

                # Evaluate the mode j
                fXj = evaluate(j, alphas, Y, dataX, b)
                Ej = fXj - float(Y[j])

                # Copy alphas
                alpha_old_i = alphas[i].copy()
                alpha_old_j = alphas[j].copy()

                # Check how much we can change the alphas
                # L = Lower bound
                # H = Higher bound
                if Y[i] != Y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # If the two are equal, directly go to next iteration
                if L == H:
                    continue

                # Calculate ETA
                eta = 2.0 * dataX[i, :] * dataX[j, :].T - \
                    dataX[i, :] * dataX[i, :].T - \
                    dataX[j, :] * dataX[j, :].T
                
                # if eta is greater or equal to zero, 
                #directly go to next iteration
                if eta >= 0:
                    continue

                # Update J and I alphas
                alphas[j] -= Y[j] * (Ei - Ej) / eta
                alphas[j] = clipalpha(alphas[j], L, H)
                # If alpha is not moving enough, continue..
                if abs(alphas[j] - alpha_old_j) < 0.00001:
                    continue
                # Change alpha_i for the exact value, in the opposite
                # direction
                alphas[i] += Y[j] * Y[i] * (alpha_old_j - alphas[j])

                # Update b
                b1 = b + Ei + Y[i] * (alphas[i] - alpha_old_i) * \
                    dataX[i, :] * dataX[i, :].T + \
                    Y[j] * (alphas[j]-alpha_old_j) * \
                    dataX[i, :] * dataX[j, :].T

                b2 = b + Ej + Y[i] * (alphas[i] - alpha_old_i) * \
                    dataX[i, :] * dataX[j, :].T + \
                    Y[j] * (alphas[j]-alpha_old_j) * \
                    dataX[j, :] * dataX[j, :].T

                # Choose b 
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # if the iteration can get this stage, then passes+=1
                # which actually calcuate the effective iteration
                passes += 1
                tmp = dualObjective(alphas, Y, dataX)
                dualmaximum.append(float(tmp))
               
                

        dist = np.linalg.norm(alphas_old-alphas)
        if (dist <= 0.001):
            break
    return dualmaximum

times = []
save_path = './'

for i in range(int(sys.argv[2])):
    start_time = time.time()
    result = smoSimple()
    times.append(time.time()-start_time)
    result = np.array(result)
    np.savetxt(save_path+"RihanSMO-"+str(i)+".txt", result)


times = np.array(times)
print("Avg runtime: " + str(np.average(times)) + "s")
print("Std runtime: " + str(np.std(times)) + "s")
print("Plot data have been plolted in ./RihanSMO-k.txt")




