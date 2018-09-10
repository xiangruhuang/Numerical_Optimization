This is a set of functions that implement optimization to find a low-rank matrix approximation. 

A simple demonstration of how to call the functions can be found in demo.m file, where we demonstrate the performance of optimization algorithms (trust region method and alternating minimization).

With the input data matrix A, a sparse observation pattern G, a small constant \mu, a initialization for B and C(B_0, C_0), the algorithm will return a approximating local minimum. Also, the objective function evaluated at the 'optimal' point and the number of step required to find the optimal point will also be returned. The mathematics behind the codes is shown in a separated write-up submitted with the theory part.
