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

%% Alternating Minimization Method
delta_k = 0.5; % Initial trust region
delta_hat = 1;
eta = 0.2;
numEpochs = 20;
objective_vals = [];

for k=1:1:numEpochs
    % Using Cauchy point to get the current pk
    [pk, gk, bk] = calculatePk(A, G, B, C, mu, n, r, delta_k); 
    fk = objective( A, G, B, C, mu, n );
    objective_vals(k) = fk;
    rhok = ( fk - objective( A, G, B + reshape(pk(1:n*r), [n r]), ...
           C + reshape(pk(n*r+1:2*n*r), [r n]), mu, n )) / ...
           ( fk - subProblem(fk, pk, gk, bk));
    if rhok < 1/4
        delta_k = 1/4*delta_k;
    else
        if rhok > 3/4 && norm(pk) == delta_k
            delta_k = min(2*delta_k, delta_hat);
        end
    end
    if rhok >= eta
        B = B + reshape(pk(1:n*r), [n r]);
        C = C + reshape(pk(n*r+1:2*n*r), [r n]);
    end
end
plot(objective_vals);