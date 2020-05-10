import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt('images2.csv', delimiter=',', skiprows=1)
train_x = train[:,0:2]
train_y = train[:,2]

theta = np.random.rand(3)

# 標準化(統計)
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)
def standardize(x):
    return (x - mu) / sigma
train_z = standardize(train_x)

def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, x])
X = to_matrix(train_z)

def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

ETA = 1e-3

epoch = 5000

for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)

x0 = np.linspace(-2, 2, 100)
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x0, -(theta[0] + theta[1] * x0) / theta[2], linestyle='dashed')
plt.show()
