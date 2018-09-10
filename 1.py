import matplotlib.pyplot as plt
import pdb
from math import sqrt
import numpy as np
from numpy.linalg import norm, inv
from numpy.random import randint
from numpy.matlib import zeros
from numpy.matlib import rand


print "Generating testing data: A and G"

n = 20
r = 5
G = np.matrix(randint(2, size=(n,n)))
true_B = np.matrix(randint(10, size=(n,r)))
true_C = np.matrix(randint(10, size=(r,n)))
A = true_B*true_C
mu = 0.00001


def loss(B, C):
    # B and C should be numpy matrices
    result = norm(np.multiply(G, (A - B*C)), 'fro')
    reg = norm(B, 'fro') + norm(C, 'fro')
    result += mu/2.0 * reg
    return result


def f(xk):
    B = xk[0:n*r,0].reshape(n,r)
    C = xk[n*r:2*n*r,0].reshape(r,n)
    return loss(B,C)

def grad_B(B, C):
    result = zeros((n,r))
    for i in range(n):
        for j in range(r):
            tmp = A[i,:] - B[i,:]*C
            tmp = np.multiply(tmp, G[i,:])
            tmp = np.multiply(tmp, C[j,:])
            result[i,j] = -np.sum(tmp)
    result += mu*B
    return result


def grad_C(B, C):
    result = zeros((r,n))
    for i in range(r):
        for j in range(n):
            tmp = A[:,j] - B*C[:,j]
            tmp = np.multiply(tmp, G[:,j])
            tmp = np.multiply(tmp, B[:,i])
            result[i,j] = -np.sum(tmp)
    result += mu*C
    return result


def alternating_min(T=1000):
    B = np.matrix(rand(n,r))
    C = np.matrix(rand(r,n))
    eta_C=0.0001
    eta_B=0.0001
    err = []
    for t in range(T):
        B -= eta_B * grad_B(B,C)
        C -= eta_C * grad_C(B,C)
        err.append(loss(B,C))
    return B, C, err


def grad(xk):
    B = xk[0:n*r,0].reshape(n,r)
    C = xk[n*r:2*n*r,0].reshape(r,n)
    tmp1 = grad_B(B,C).reshape(n*r,1)
    tmp2 = grad_C(B,C).reshape(n*r,1)
    return np.concatenate((tmp1, tmp2))


def trust_region_bfgs(T=1000, delta_hat=1, delta_0=0.1, eta=0.00001):
    xk = np.matrix(rand(2*n*r,1))
    Bk = np.matrix(np.identity(2*n*r))
    delta_k = delta_0
    err = []
    for i in xrange(T):
        f_k = f(xk)
        g_k = grad(xk)

        pks = - delta_k * g_k / norm(g_k,2)
        gBg = (g_k.T * Bk * g_k)[0,0]
        if gBg <= 0:
            tau_k = 1
        else:
            tau_k = min(1, norm(g_k,2)**3/(delta_k*gBg))
        pk = tau_k * pks

        # Update radius
        rou_k = - (f_k - f(xk + pk)) / (g_k.T*pk + 0.5*pk.T*Bk*pk)
        rou_k = rou_k[0,0]
        if rou_k < 0.0001:
            delta_k *= 0.25
        else:
            if rou_k > 0.75 and norm(pk,2) == delta_k:
                delta_k = min(2*delta_k, delta_hat)
            else:
                delta_k = delta_k

        if rou_k > eta:
            xkp1 = xk + pk
            # Update Bk
            sk = xkp1 - xk
            yk = grad(xkp1) - g_k
            Bk += (yk*yk.T)/(yk.T*sk) - Bk*sk*sk.T*Bk/(sk.T*Bk*sk)
            xk = xkp1
        else:
            xk = xk

        err.append(f(xk))

    return xk, err



B, C, err = alternating_min()
print err
xk, err = trust_region_bfgs()
print err
plt.semilogy(err, label='err')
plt.show()
pdb.set_trace()
