import numpy as np
from numpy.random import multivariate_normal as mvnorm
from numpy.random import choice
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import neighbors

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
                           cov=[[1/25, 0], [0, 1/25]],
                           size=1)
    dataTrain[i+100,] = mvnorm(mean=tmpMuOrange,
                               cov=[[1/25, 0], [0, 1/25]],
                               size=1)
np.var(dataTrain[:,0])
np.var(dataTrain[:,1])
np.mean(dataTrain[:,0])
np.mean(dataTrain[:,1])

dataTrain = np.concatenate([dataTrain,
                            np.repeat([[0], [1]], 100, axis=0)],
                           axis=1)
grpBlue = dataTrain[:,2] == 0
grpOrange = dataTrain[:,2] == 1
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
reg1.intercept_
reg1.coef_
reg1_pred = reg1.predict(dataTrain[:, 0:2])
reg1_line = (0.5 - reg1.intercept_ - dataTrain[:, 0]*reg1.coef_[0,0])/ \
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
x1_pred = np.linspace(-3, 3, 10000)
x1_pred.shape
knn1_fit = knn1.fit(X=dataTrain[:, 0:2], y=dataTrain[:, 2:3])

knn1_fit.predict()


x_tmp = np.array([np.repeat(x1_pred[0], 10000), x1_pred]).T
x_tmp.shape
knn1_fit.predict(x_tmp)


np.repeat([x1_pred[0], x1_pred], 100, axis=0)],
np.concatenate(x1_pred[0], x1_pred)


knn1_fit.predict(xvals_pred[0][:,1])




xvals_pred = np.meshgrid(np.linspace(-3, 3, 1000),
                         np.linspace(-3, 3, 1000)).T.reshape(-1,2)
type(xvals_pred)
type(xvals_pred[0])
xvals_pred[0].shape


xvals_pred.reshape(-1, )



np.apply_along_axis(knn1_fit.predict, axis=1, arr=xvals_pred[0])


knn1_pred = knn1.fit(X=dataTrain[:, 0:2],
                     y=dataTrain[:, 2:3]).predict(xvals_pred)

## here find which values are closest to 0.5
diff = knn1_pred - 0.5
smallDiff = np.logical_and(-.1 < diff, diff < .1)
smallDiff[:,0].shape
np.unique(smallDiff, return_counts=True)


plt.figure()
plt.scatter(dataTrain[grpBlue, 0], dataTrain[grpBlue, 1],
            facecolors='none', edgecolors='blue')
plt.scatter(dataTrain[grpOrange, 0], dataTrain[grpOrange, 1],
            facecolors='none', edgecolors='orange')
plt.scatter(x12pred[smallDiff[:, 0], 0:1], x12pred[smallDiff[:, 0], 1:2])
#plt.plot(x12pred[smallDiff[:, 0],0:1], x12pred[smallDiff[:, 0], 1])
#plt.scatter(x12pred[:, 0:1], x12pred[:, 1:2], color='grey')
plt.axis([-3, 3, -3, 3])
plt.show()
plt.close('all')
