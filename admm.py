import numpy as np
import numpy.linalg as LA
import pandas as pd

def soft_threshold(y, lam):
	s = np.zeros_like(y)
	s[y > lam] = y[y > lam] - lam
	s[y < -lam] = y[y < -lam] + lam
	return s

N = 1000
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

#L=np.linalg.norm(A.T@A,ord="fro")/lam
#L = np.max(np.sum(np.abs(A), axis=0))

lam = 0.2
mu = 0.1

x = A.T @ y / N
z = x.copy()
h = np.zeros_like(x)
N = A.shape[0]
M = A.shape[1]

for i in range(3000):
	x = LA.inv(mu * np.identity(M) + A.T @ A / N) @ (A.T @ y / N+ mu * z - h)
	z = soft_threshold(x + h / mu, lam / mu)
	h = h + mu * (x - z)
print(z.reshape(len(z)))

