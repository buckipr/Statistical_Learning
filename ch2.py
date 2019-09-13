import numpy as np
from numpy.random import multivariate_normal as mvnorm
from numpy.random import choice
import matplotlib.pyplot as plt

# Example in Section 2.3
muBlue = mvnorm(mean=[1,0], cov=[[1,0], [0,1]], size=10)
muOrange = mvnorm(mean=[0,1], cov=[[1,0], [0,1]], size=10)

# sampling
dataTrain = np.zeros(shape=(200,2))
for i in range(0,100): # i = 0
    rowMuBlue = choice(a=10, size=1, replace=True, p=np.full(10, 1 / 10))
    rowMuOrange = choice(a=10, size=1, replace=True, p=np.full(10, 1 / 10))
    tmpMuBlue = muBlue[rowMuBlue].tolist()[0]
    tmpMuOrange = muOrange[rowMuOrange].tolist()[0]
    dataTrain[i,] = mvnorm(mean=tmpMuBlue,
                           cov=[[1/5, 0], [0, 1/5]],
                           size=1)
    dataTrain[i+100,] = mvnorm(mean=tmpMuOrange,
                               cov=[[1/5, 0], [0, 1/5]],
                               size=1)

dataTrain = np.concatenate([dataTrain,
                            np.repeat([[0], [1]], 100, axis=0)],
                           axis=1)

grpBlue = dataTrain[:,2] == 0
grpOrange = dataTrain[:,2] == 1
plt.scatter(dataTrain[grpBlue, 0], dataTrain[grpBlue, 1],
            facecolors='none', edgecolors='blue')
plt.scatter(dataTrain[grpOrange, 0], dataTrain[grpOrange, 1],
            facecolors='none', edgecolors='orange')

