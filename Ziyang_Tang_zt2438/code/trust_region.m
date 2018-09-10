function [ B,C,otrial ] = trust_region( G,A,B0,C0,mu,maxIter )
%TRUST_REGION Summary of this function goes here
%   Detailed explanation goes here
B = B0;
C = C0;
[n,r] = size(B);
objective = norm(G.*(A-B*C),'fro')^2 + mu/2*(norm(B,'fro')+norm(C,'fro'));
otrial = objective;
delta = 10;     % Trust radius

for i = 2:maxIter
    fi = otrial(end);
    deltaB = -2*((G).*(A-B*C))*C' + mu*B;
    deltaC = -2*B'*((G).*(A-B*C)) + mu*C;
    gi = [reshape(deltaB',n*r,1);reshape(deltaC,n*r,1)];    %gradient
    Hi = zeros(n*r*2,n*r*2);    % Hessian
    % Calculate Hessian
    for j = 0:(n-1)
        jB = j*r+1:(j+1)*r;
        Hi(jB,jB) = 2*bsxfun(@times,G(j+1,:),C)*C' + mu*eye(r);
        Hi(jB+n*r,jB+n*r) = 2*bsxfun(@times,G(:,j+1)',B')*B+mu*eye(r);
        for k = 0:(n-1)
            kC = (k+n)*r+1:(k+n+1)*r;
            Hi(jB,kC) = 2*G(j+1,k+1)*(B(j+1,:)'*C(:,k+1)' + (B(j+1,:)*C(:,k+1)-A(j+1,k+1))*eye(r));
            Hi(kC,jB) = Hi(jB,kC)';
        end
    end
    
    L = -eigs(Hi,1,'SA');   % Smallest lambda that can achieve positive semidefinite
    R = 500;                % Give a large value as lower bound
    pi = -(Hi+(L+R)/2*eye(n*r*2))\gi;
    % Do binary search to find best lambda
    % Option solution, Use Newton methods.
    while abs(norm(pi) - delta) > 0.01
        if norm(pi) > delta
            L = (L+R)/2;
        else
            R = (L+R)/2;
        end
        if R-L < 0.001 
            break;
        end
        pi = -(Hi+(L+R)/2*eye(n*r*2))\gi;
    end
    deltaB = reshape(pi(1:n*r),r,n)';
    deltaC = reshape(pi(n*r+1:2*n*r),r,n);
    Bnew = B + deltaB;
    Cnew = C + deltaC;
    fnew = norm(G.*(A-Bnew*Cnew),'fro')^2 + mu/2*(norm(Bnew,'fro')+norm(Cnew,'fro'));
    rho = (fi-fnew)/(-gi'*pi-0.5*pi'*Hi*pi);
    
    if rho < 0.25
        delta = delta * 0.25;
    else
        if rho > 0.75
            delta = min(40,2*delta);
        end
    end
    
    if rho > 0.2
        B = Bnew;
        C = Cnew;
        otrial = [otrial,fnew];
    else
        otrial = [otrial,fi];
    end
end

end

