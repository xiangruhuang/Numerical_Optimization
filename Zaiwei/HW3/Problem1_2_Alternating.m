%% Initial Setup
clear;
n = 10; % Data Matrix Dimensionality
r = 5; % Dimension r for B and C
mu = 0.1;
sparsity = 0.2; % Probability that 1 appears in each cell
A = random('normal',0,1,[n n]); % Data Matrix
G = generateSparseM(n, sparsity); % Sparse observation pattern
B = random('normal',0,0.02,[n r]); % Gaussian random noise initialization on B 
C = random('normal',0,0.02,[r n]); % Gaussian random noise initialization on C

%% Trust Region Method
delta_k = 0.5; % Initial trust region
delta_hat = 1;
eta = 0.2;
numEpochs = 20;
objective_vals = [];

for k=1:1:numEpochs
    fk = objective( A, G, B, C, mu, n );
    objective_vals(k) = fk;
    % First fix C, minimize B
    [LHS, RHS] = fixC(A, G, C, mu, n, r); 
    B = reshape(linsolve(LHS, RHS), [r n])';
    % Then fix B, minimize C
    [LHS, RHS] = fixB(A, G, B, mu, n, r);
    C = reshape(linsolve(LHS, RHS), [r n]);
end
plot(objective_vals);