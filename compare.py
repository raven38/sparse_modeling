import numpy as np
import numpy.linalg as LA
import pandas as pd
from sklearn.linear_model import  Lasso

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

lam = 200
L = LA.eig(A.T@A)[0].max() 

eps = 1e-8
num_iter = 5000

xt = A.T @ y / N
old_beta = 0
wt = xt

losses = []
pars = []

for i in range(num_iter):
	losses.append(np.sum((y - A @ xt)**2) / 1000)
	pars.append(xt)
	res = y - A @ wt
	v = wt + (A.T @ res) / L
	new_xt = soft_threshold(v, lam/L)
	new_beta = (1 + np.sqrt(1 + 4 * old_beta * old_beta)) / 2
	wt = new_xt + (old_beta - 1) / new_beta * (new_xt - xt)
	xt = new_xt
	old_beta = new_beta

np.savetxt("fista_loss.csv", np.array(losses), delimiter=',')
np.savetxt("fista_parameter.csv", np.array(pars).reshape(num_iter, 5), delimiter=',')

print("fista solution: ", xt.reshape(1, 5))
print("fista's MSE: ", np.sum((y - A @ xt)**2) / 1000)

lam = 0.2
mu = 0.1

x = A.T @ y / N
z = x.copy()
h = np.zeros_like(x)
N = A.shape[0]
M = A.shape[1]

losses = []
pars = []

for i in range(num_iter):
	losses.append(np.sum((y - A @ x)**2) / 1000)
	pars.append(x)
	x = LA.inv(mu * np.identity(M) + A.T @ A / N) @ (A.T @ y / N+ mu * z - h)
	z = soft_threshold(x + h / mu, lam / mu)
	h = h + mu * (x - z)
	
np.savetxt("admm_loss.csv", np.array(losses), delimiter=',')
np.savetxt("admm_parameter.csv", np.array(pars).reshape(num_iter, 5), delimiter=',')

print("ADMM solution: ", z.reshape(1, 5))
print("ADMM's MSE: ", np.sum((y - A @ x)**2) / 1000)

lasso = Lasso(alpha=0.2).fit(A, y)

print("Scikit Learn solution: ", lasso.coef_)
print("Sciket Learn MSE: ", np.sum((y - A @ lasso.coef_.reshape(5, 1))**2) / 1000)

x =  np.array([4, -2, 0, 0, 1])
print("[4, -2, 0, 0, 1]: ", x)
print("[4, -2, 0, 0, 1]'s MSE: ", np.sum((y - A @ x)**2) / 1000)

