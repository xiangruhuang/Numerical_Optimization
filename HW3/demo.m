%%% Data Generation
%% hyperparameters
n = 10; % dimension of original matrices
r = 5; % low rank 
mu = 0.1; %small cosntant
density = 0.1; % sparsity parameter
G = sprand(n,n,density)>0; % sparse matrix
A = randn(n,n); % data matrix
%%% Optimization
% intial point for B AND C
B_0 = randn(n,r);
C_0 = randn(r,n);
%%  Trust Region
[B_t, C_t, obj_t, step_t] = trust_region(A, G, mu, r, B_0, C_0); 
obj_t %objective function evaluated at the point B_t, C_t
step_t % number of steps required to get to the optimal point
B_t % optimal for B found by trust region
C_t % optimal for B found by trust region
%% Alternating minimization
[B_a, C_a, obj_a, step_a] = alter_min(A, G, mu, r, B_0, C_0); 
obj_a %objective function evaluated at the point B_a, C_a
step_a % number of steps required to get to the optimal point
B_a % optimal for B found by alternating minimization
C_a % optimal for C found by alternating minimization



