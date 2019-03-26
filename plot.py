import matplotlib.pyplot as plt.plot
import numpy as np

f_loss = np.loadtxt('fista_loss.csv')
a_loss = np.loadtxt('admm_loss.csv')
f_par = np.loadtxt('fista_parameter.csv', delimiter=',')
a_par = np.loadtxt('admm_parameter.csv', delimiter=',')

fig = plt.Figure()

plt.yscale("log")
plt.plot(f_loss[:100])
plt.plot(a_loss[:100])
plt.legend(['FISTA', 'ADMM'])
plt.title('comparision loss of each algorithms')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.close()

fig = plt.Figure()
plt.yscale("symlog")
plt.plot(a_par[:15])
plt.legend(['x1', 'x2', 'x3', 'x4','x5'])
plt.title('Magnitude of the coefficients on ADMM')
plt.xlabel('Iterations')
plt.ylabel('Magnitude')
plt.savefig('admm_parameters.png')
plt.close()

fig = plt.Figure()
plt.yscale("symlog")
plt.plot(f_par[:500])
plt.legend(['x1', 'x2', 'x3', 'x4','x5'])
plt.title('Magnitude of the coefficients on FISTA')
plt.xlabel('Iterations')
plt.ylabel('Magnitude')
plt.savefig('fista_parameters.png')
plt.close()
