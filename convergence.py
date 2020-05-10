import numpy as np
import matplotlib.pyplot as plt

# データセット
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:,0]
train_y = train[:,1]

# グラフ標準化(データ-平均/標準偏差)
mu = train_x.mean()
sigma = train_x.std()

def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T

X = to_matrix(train_z)
print(X)

# パラメータ
theta = np.random.rand(3)
print(theta)

def f(x):
    return np.dot(x, theta)

def E(x,y):
    return 0.5 * np.sum((y - f(x)) ** 2)

def EMS(x, y):
    return (1/ x.shape[0]) * np.sum((y - f(x)) ** 2)

# エラー
errors = []

# 学習率
ETA = 1e-3

# 誤差の差分
diff = 1

errors.append(EMS(X, train_y))
# error = E(X, train_y)
while diff > 1e-2:
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    errors.append(EMS(X, train_y))
    diff = errors[-2] - errors[-1]

x = np.arange(len(errors))
plt.plot(x, errors)
plt.show()
