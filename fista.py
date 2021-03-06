import numpy as np
import numpy.linalg as LA
import pandas as pd
from sklean.linear_model import  Lasso

def soft_threshold(y, lam):
	s = np.zeros_like(y)
	s[y > lam] = y[y > lam] - lam
	s[y < -lam] = y[y < -lam] + lam
	return s

N = 100
df = pd.read_csv('data.csv')
y = np.array(df['y']).reshape(-1, 1)
y = (y - y.mean()) # / y.std()
A = np.array(df[['x1', 'x2', 'x3', 'x4', 'x5']])
A[:, 1:4] = (A[:, 1:4] - A[:, 1:4].mean(axis=0)) / A[:, 1:4].std(axis=0)

# n_samples, n_features = 50, 200
# A = np.random.randn(n_samples, n_features)
# coef = 3 * np.random.randn(n_features)
# inds = np.arange(n_features)
# np.random.shuffle(inds)
# coef[inds[10:]] = 0  # sparsify coef
# y = np.dot(A, coef)

# # add noise
# y += 0.01 * np.random.normal(size=n_samples)

lam = 200
L = 2*LA.eig(A.T@A)[0].max() 

eps = 1e-8

xt = A.T @ y
old_beta = 0
wt = xt

for i in range(100):
	res = y - A @ wt
	if np.linalg.norm(res, 2) < eps:
		break
	v = wt + (A.T @ res) / L
	new_xt = soft_threshold(v, lam/L)
	new_beta = (1 + np.sqrt(1 + 4 * old_beta * old_beta)) / 2
	wt = new_xt + (old_beta - 1) / new_beta * (new_xt - xt)
	xt = new_xt
	old_beta = new_beta
		
print(xt)
