import numpy

lamb = 1
eps = 1e-12

class Segment(object):
    """ 
        suml: number of points to the left of this segment
        sumr: number of points to the right of this segment
        count: number of points contained in this segment
    """
    def __init__(self, u, x_mean, suml, sumr, count):
        self.u = u
        self.x_mean = x_mean
        self.suml = suml
        self.sumr = sumr
        self.count = count
    
    def to_list(self):
        N = int(numpy.round(self.count))
        list_u = [self.u] * N
        list_count = [self.count] * N
        return list_u, list_count

    """ Merge with another segment `seg`
        Assume `self` is the left one and `seg` is the right one

        lamb: lambda
    """
    def merge(self, seg):
        global lamb
        count = self.count + seg.count
        x_mean = (self.x_mean * self.count + seg.x_mean * seg.count)/count
        suml = self.suml
        sumr = seg.sumr
        u = x_mean + (sumr - suml) * lamb
        return Segment(u, x_mean, suml, sumr, count)

    def __str__(self):
        return 'u=%f, x=%f, suml=%f, sumr=%f, count=%f' % (self.u, self.x_mean,
                self.suml, self.sumr, self.count)

"""
    Try merging the top and the second top segments in this stack
    the top segment is the rightmost one
    stack: a list of segments viewed as a stack
"""
def down_merge(stack):
    global eps
    while len(stack) > 1:
        segr = stack[-1]
        segl= stack[-2]
        if segr.u - eps < segl.u:
            """ Merge seg1 and seg2 """
            new_seg = segl.merge(segr)
            stack.pop()
            stack[-1] = new_seg
        else:
            break
    return stack

def check(x, u, l):
    global lamb, eps
    N = len(x)

    suml = 0
    sumr = N
    for i in range(N):
        if (i == 0) or abs(u[i] - u[i-1]) > eps:
            sumr -= l[i]
        c = u[i] - x[i] - (sumr - suml)*lamb
        if (c - l[i]*lamb > eps) or (c + l[i]*lamb < -eps):
            print('x', x)
            print('u', u)
            print('l', l)
            print('i=%d, suml=%f, sumr=%f, c=%f, lamb=%f, mean_x=%f' % (i, suml, sumr, c,
                lamb, numpy.mean(x)))
            return False
        if (i == N-1) or abs(u[i+1] - u[i]) > eps:
            suml += l[i]

    return True

"""
    x: a vector of dimension N, assuming sorted non-decreasingly
    output: u_star = argmin_{u} \sum_{i < j} |u_i - u_j| +  0.5 * \sum_i (u_i - x_i)^2
"""
def prox_1D(x):
    global lamb

    """ for recovery """
    x0 = [xi for xi in x]
    N0 = len(x)

    N = N0
    stack = []
    for i in range(N):
        u = x[i] + (N-1-2*i)*lamb
        seg = Segment(u, x[i], i, N-1-i, 1.0)
        stack.append(seg)
        down_merge(stack)

    u_star = []
    l_star = []
    for seg in stack:
        u_s, l_s = seg.to_list()
        u_star = u_star + u_s
        l_star = l_star + l_s

    assert check(x0, u_star, l_star)
    return u_star

""" 
    x: M vectors, each has dimension N
    output: u_star = argmin_{u} lamb * \sum_{i < j} \|u_i - u_j \|_1 +  0.5 * \sum_i \|u_i - x_i\|_2^2
"""
def prox(x):
    [M, N] = x.shape
    u = []
    for n in range(N):
        z = numpy.asarray([x[i][n] for i in range(M)])
        sort_index = numpy.argsort(z)
        sorted_u = prox_1D([z[sort_index[i]] for i in range(M)])
        u_d = [0] * M
        for i in range(M):
            u_d[sort_index[i]] = sorted_u[i]
        u.append(u_d)
    u = numpy.transpose(numpy.asarray(u))
    return u 

import scipy.io as sio

data = sio.loadmat('hw4_data.mat')
A = []
for Ai in data['A']:
    A.append(Ai[0])

b = []
for bi in data['b']:
    b.append(numpy.squeeze(bi[0]))

M = len(A)
assert M == len(b)
N = A[0].shape[1]

#print A[0].shape, b[0].shape

x = numpy.zeros([M, N])

z = numpy.copy(x)

max_iter = 1000
alpha = 1e-3
beta = 1.0
for i in range(max_iter):
    grad = numpy.zeros([M, N])
    for m in range(M):
        #print A[m].shape, x[m].shape, b[m].shape
        #print type(A[m]), type(x[m]), type(b[m])
        #temp = A[m].dot(x[m]) 
        #print temp.shape
        grad[m] = 2*A[m].T.dot(A[m].dot(z[m]) - b[m])
    #print x.shape
    lamb = lamb*alpha
    beta_new = 0.5*(1 + numpy.sqrt(1 + 4.0*beta*beta))
    x_new = prox(z - alpha*grad)
    z_new = x_new + (beta - 1)/ (beta_new)*(x_new - x)
    x = x_new
    z = z_new
    beta = beta_new
    #x = x - t_i*grad
    lamb = lamb/alpha
    #print x.shape
    obj = 0.0
    for m in range(M):
        obj += numpy.linalg.norm(A[m].dot(x[m]) - b[m], 2) ** 2

    for m in range(M):
        for m2 in range(m):
            """ m2 < m """
            obj += lamb*numpy.linalg.norm(x[m2] - x[m], 1)
    print('iter=%d, obj=%f' % (i, obj))

#""" Construct D by N matrix `x` """
#x = []
#for d in range(D):
#    x.append([numpy.random.uniform()*N for n in range(N)])
#x = numpy.asarray(x)
#
#print(x.shape)
#prox(x)
