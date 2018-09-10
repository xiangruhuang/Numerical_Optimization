%% generate sparse matrix
clear
clc
rng(88)
n = 10; % dimension of matrix
r = 5; 
sparsity = 0.05; % set the percentage of sparsity of G
idx_G = randsample(n * n, n * n * sparsity);
G = zeros(n, n); % not necessary, can optimize to use idx only
G(idx_G) = 1;
A = rand(n, n);
mu = 0.05;

tic
[ite_tr, opt_tr_point, opt_tr_val] = trust_region(n, r, G, A,...
    mu, @gradient, @Hessian, @f);
toc

tic
[ite_ad, opt_ad_point, opt_ad_val] = alter_direction(n, r, G, A,...
    mu, @f, @gradient);
toc

opt_tr_val
opt_ad_val

[opt_tr_B, opt_tr_C] = recover_mat(opt_tr_point, n, r);
[opt_ad_B, opt_ad_C] = recover_mat(opt_ad_point, n, r);