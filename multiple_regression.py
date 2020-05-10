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

# 学習率
ETA = 1e-3

# 誤差の差分
diff = 1

error = E(X, train_y)
while diff > 1e-2:
    theta = theta - ETA * np.dot(f(X) - train_y, X)

    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()
