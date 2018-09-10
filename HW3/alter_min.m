function [B, C, obj, step] = alter_min(A, G, mu, r, B_0, C_0)
%ALTER_MIN Summary of this function goes here
%   Fixed one matrix B/C, we optimize the other one with line search
%   algorithm. We use back tracking to find the appropriate line length. 
%% parameters 
stop_sign = 10^-6; c = 10^(-4); rho = 0.1;
%% initialization
[n,~] = size(A);
B = B_0;
C = C_0;
g = derivative(B, C, A, G, mu); 
%% iteration
step = 0;
while(norm(g)> stop_sign)
    % optimize B
    g = derivative(B, C, A, G, mu); 
    p = -g(1:n*r);
    dB = reshape(p,n,r);    
    alpha = 1;
    while objective(B+alpha*dB,C,A,G,mu)>(objective(B,C,A,G,mu)-c*alpha*norm(p)^2)
        alpha = rho * alpha;
    end
    B = B + alpha*dB;
    % optimize C
    g = derivative(B, C, A, G, mu); 
    q = -g((n*r+1):end);
    dC = reshape(q,r,n);    
    alpha = 1;
    while objective(B,C+alpha*dC,A,G,mu)>(objective(B,C,A,G,mu)-c*alpha*norm(q)^2)
        alpha = rho * alpha;
    end
    C = C + alpha*dC;
    step = step + 1;
%     step
%     format long
%     objective(B,C,A,G,mu)
end
obj = objective(B,C,A,G,mu);
end

