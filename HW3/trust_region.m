function [B, C, obj, step] = trust_region(A, G, mu, r, B_0, C_0)
%TRUST_REGION Summary of this function goes here
%   Detailed explanation goes here
%% parameters 
delta_hat = 1;
ita = 1/5;
stop_sign = 10^-6;
%% initialization
[n,~] = size(A);
B = B_0;
C = C_0;
delta = delta_hat;
g = derivative(B, C, A, G, mu); % derivative at initial point
% f = objective(B, C, A, G, mu); % obj a initial point
H = speye(n*r*2); % intial approximation for hessian (B_0), BFGS update is used here
%% iteration
step = 0;
while(norm(g)> stop_sign)
    if g*H*g'<=0
        p = -delta/norm(g)*g;
    else
        p = -min(1,norm(g)^3/(delta*g*H*g'))*delta/norm(g)*g;
    end
    dB = reshape(p(1:n*r),n,r);
    dC = reshape(p((n*r+1):end),r,n);
    rho = (objective(B,C,A,G,mu)-objective(B+dB,C+dC,A,G,mu))/(-g*p'-1/2*p*H*p');
    if rho<1/4
        delta = 1/4*delta;
    else
        if rho>3/4 && norm(p)==delta
            delta = min(2*delta, delta_hat);
        end
    end
    if rho>ita
        B_new = B + dB;
        C_new = C + dC;
        g_new = derivative(B_new, C_new, A, G, mu);
        y = g_new - g;
        s = p;
        H = H + y'*y/(y*s') - (H*s'*s*H)/(s*H*s'); 
        B = B_new;
        C = C_new;
        g = g_new;
        step = step + 1;
%        step
%        format long
%        objective(B,C,A,G,mu)
    end
end
obj = objective(B,C,A,G,mu);
end

