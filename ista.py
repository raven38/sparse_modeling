import numpy as np
import numpy.linalg as LA
import pandas as pd

def soft_threshold(y, lam):
	s = np.zeros_like(y)
	s[y > lam] = y[y > lam] - lam
	s[y < -lam] = y[y < -lam] + lam
	return s

N = 100
df = pd.read_csv('data.csv')
df_x1 = pd.get_dummies(df['x1'], drop_first=True)
df_x5 = pd.get_dummies(df['x5'], drop_first=True)
y = np.array(df['y']).reshape(-1, 1)
y = (y - y.mean()) / y.std()
A = df[['x2', 'x3', 'x4']]
A = (A - A.mean(axis=0)) / A.std(axis=0)
A = np.hstack((A, df_x1, df_x5))

# n_samples, n_features = 50, 200
# A = np.random.randn(n_samples, n_features)
# coef = 3 * np.random.randn(n_features)
# inds = np.arange(n_features)
# np.random.shuffle(inds)
# coef[inds[10:]] = 0  # sparsify coef
# y = np.dot(A, coef)

# # add noise
# y += 0.01 * np.random.normal(size=n_samples)

lam = 1e-10
L = LA.eig(A.T@A)[0].max()

eps = 1e-8

xt = np.random.rand(45, 1) - 0.5

for i in range(5000):
	res = y - A @ xt
	if np.linalg.norm(res, 2) < eps:
		break
	v = xt + (A.T @ res) / L
	xt = soft_threshold(v, 1/L)

print(xt)
