%% Parameter setting
n = 100;
r = 10;
mu = 1;
epsilon = 0.01;
maxIter = 20;

%% Random data generating
G = sprand(n,n,0.01,1);
A = randn(n);

B = randn(n,r);
C = randn(r,n);

%% Begin optimizing and recording
objective = norm(G.*(A-B*C),'fro')^2 + mu/2*(norm(B,'fro')+norm(C,'fro'));
otrial = objective;
otrial2 = objective;
B2 = B;
C2 = C;

% Trust Region
[B3,C3,otrial3] = trust_region(G,A,B,C,mu,maxIter);

for i = 2:maxIter
    % Gradient Descend
    deltaB = -2*((G).*(A-B*C))*C' + mu*B;
    deltaC = -2*B'*((G).*(A-B*C)) + mu*C;
    C = C- epsilon*deltaC;
    B = B - epsilon*deltaB;
    otrial = [otrial,norm(G.*(A-B*C),'fro')^2+ mu/2*(norm(B,'fro')+norm(C,'fro'))];
    
    % Alternating minimization
    B3 = B2;
    C3 = C2;
    for j = 1:n
        T = bsxfun(@times,G(j,:),C2)*C2'+mu/2*eye(r);
        S = G.*A;
        S = S(j,:)*C2';
        B3(j,:) = (T\S')';  % Optmimum solution for each bj
    end
    B2 = B3;
    for j = 1:n
        T = bsxfun(@times,G(:,j)',B2')*B2+mu/2*eye(r);
        S = G.*A;
        S = S(:,j)'*B2;
        C3(:,j) = T\S';     % Optmimum solution for each cj
    end
    C2 = C3;
    otrial2 = [otrial2,norm(G.*(A-B2*C2),'fro')^2+mu/2*(norm(B2,'fro')+norm(C2,'fro'))];
end

plot(1:maxIter,otrial,'b',1:maxIter,otrial2,'r',1:maxIter,otrial3,'g');