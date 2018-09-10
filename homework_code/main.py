import scipy.io as sio
import numpy as np
oo = 1e10
nIters = 20000
tk = 0.001

def g(x):
  res = 0
  for i in range(M):
    res += ((np.dot(A[i], x[i]) - b[i]) ** 2).sum()
  return res

def dg(x):
  dg_ = np.zeros((M, N))
  for i in range(M):
    dg_[i] = 2 * np.dot((np.dot(A[i], x[i].reshape(N, 1)) - b[i].reshape(D, 1)).transpose(), A[i]) 
  return dg_

def h(x):
  res = 0
  for i in range(M):
    for j in range(i + 1, M):
      res += lamb * np.abs(x[i] - x[j]).sum()
  return res

def f(x):
  return g(x) + lamb * h(x)


def proxH(x):
  u = np.zeros(x.shape)
  for i in range(N):
    idx = np.argsort(x[:, i])
    idx_inv = np.zeros(M, dtype = np.int32)
    for j in range(M):
      idx_inv[idx[j]] = j
    x_ = np.sort(x[:, i])
    u_ = np.zeros(M)
    for j in range(M):
      u_[j] = x_[j] - tk * lamb * (2 * (j + 1) - M - 1)
    #Greedy, probably wrong
    u_ = np.sort(u_)
    u[:, i] = u_[idx_inv]
  return u

M = 10
D = 300
N = 1000
data = sio.loadmat('hw4_data.mat')
lamb = 1. #/ M

A = np.zeros((M, D, N))
b = np.zeros((M, D))
for i in range(M):
  A[i] = data['A'][i][0].copy()
  b[i] = data['b'][i][0].reshape(D).copy()

print 'A', A.shape
print 'b', b.shape

x = np.zeros((M, N))

f_0 = oo
for t in range(nIters):
  print 'Iter', t, 'loss = ', f(x), 'g = ', g(x), 'h = ', h(x)
  x = proxH(x - tk * dg(x))
