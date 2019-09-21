import numpy as np
from numpy.random import multivariate_normal as mvnorm
from numpy.random import choice
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import neighbors

# Example in Section 2.3
muBlue = mvnorm(mean=[1, 0], cov=[[1, 0], [0, 1]], size=10)
muOrange = mvnorm(mean=[0, 1], cov=[[1, 0], [0, 1]], size=10)

# sampling
dataTrain = np.zeros(shape=(200, 2))
for i in range(0, 100):
    rowMuBlue = choice(a=10, size=1, replace=True, p=np.full(10, 1/10))
    rowMuOrange = choice(a=10, size=1, replace=True, p=np.full(10, 1/10))
    tmpMuBlue = muBlue[rowMuBlue].tolist()[0]
    tmpMuOrange = muOrange[rowMuOrange].tolist()[0]
    dataTrain[i, ] = mvnorm(mean=tmpMuBlue,
                            cov=[[1/25, 0], [0, 1/25]],
                            size=1)
    dataTrain[i+100, ] = mvnorm(mean=tmpMuOrange,
                                cov=[[1/25, 0], [0, 1/25]],
                                size=1)
np.var(dataTrain[:, 0])
np.var(dataTrain[:, 1])
np.mean(dataTrain[:, 0])
np.mean(dataTrain[:, 1])

dataTrain = np.concatenate([dataTrain,
                            np.repeat([[0], [1]], 100, axis=0)],
                           axis=1)
grpBlue = dataTrain[:, 2] == 0
grpOrange = dataTrain[:, 2] == 1
plt.figure()
plt.scatter(dataTrain[grpBlue, 0], dataTrain[grpBlue, 1],
            facecolors='none', edgecolors='blue')
plt.scatter(dataTrain[grpOrange, 0], dataTrain[grpOrange, 1],
            facecolors='none', edgecolors='orange')
plt.axis([-3, 3, -3, 3])
plt.show()
plt.close('all')

# linear models and least squares
reg1 = linear_model.LinearRegression()
reg1.fit(dataTrain[:, 0:2], dataTrain[:, 2:3])
print(reg1.intercept_)
print(reg1.coef_)
reg1_pred = reg1.predict(dataTrain[:, 0:2])
reg1_line = (0.5 - reg1.intercept_ - dataTrain[:, 0]*reg1.coef_[0, 0]) / \
              reg1.coef_[0, 1]
plt.figure()
plt.scatter(dataTrain[grpBlue, 0], dataTrain[grpBlue, 1],
            facecolors='none', edgecolors='blue')
plt.scatter(dataTrain[grpOrange, 0], dataTrain[grpOrange, 1],
            facecolors='none', edgecolors='orange')
plt.plot(dataTrain[:, 0:1], reg1_line, color='black')
plt.axis([-3, 3, -3, 3])
plt.show()
plt.close('all')

# nearest-neighbor method
knn1 = neighbors.KNeighborsRegressor(n_neighbors=15, weights='uniform')
x1_pred = np.linspace(-3, 3, 2000)
knn1_fit = knn1.fit(X=dataTrain[:, 0:2], y=dataTrain[:, 2:3])
# print out nearest neighbors & prediction for a random point
knn1.kneighbors(np.array([[-1, 2]]), 15)
knn1.predict(np.array([[-1, 2]]))
knn1_fit.predict(np.array([[-1, 2]]))
# find x1,x2 coordinates where prediction is closest to 0.5
# 10,000 points takes a while (maybe split across cores?)
x_line = np.zeros(shape=(2000, 2))
for i in range(0, 2000):
    x_tmp = np.array([np.repeat(x1_pred[i], 2000), x1_pred]).T
    idx = (np.abs(knn1.predict(x_tmp) - .5)).argmin()
    x_line[i, :] = x_tmp[idx, :]

# Tried to use more evaluation points by creating 3-d array and then
# applying the predict function to each sub-array, but couldn't figure it out.
# xvals_pred = np.array(np.meshgrid(np.linspace(-1, 1, 10),
#                                   np.linspace(-1, 1, 10))).T.reshape(-1,2)
# jt = xvals_pred.reshape((10,10,2))
# jt[0,:,:]
# jt[1,:,:]
# xvals_pred = np.array(np.meshgrid(np.linspace(-3, 3, 1000),
#                                   np.linspace(-3, 3, 1000))).T.reshape(-1,2)
# xvals_pred_3d = xvals_pred.reshape((10, 10, 2))
# jt2 = np.apply_over_axes(knn1.predict, xvals_pred_3d, [1,2])

plt.figure()
plt.scatter(dataTrain[grpBlue, 0], dataTrain[grpBlue, 1],
            facecolors='none', edgecolors='blue')
plt.scatter(dataTrain[grpOrange, 0], dataTrain[grpOrange, 1],
            facecolors='none', edgecolors='orange')
plt.plot(x_line[:, 0], x_line[:, 1])
plt.axis([-3, 3, -3, 3])
plt.show()
plt.close('all')
