function [ h ] = secondOrder( A, G, B, C, mu, n, r )
% This function calculates the second order gradient (hessian) of the 
% objective function.
hbb = mu * eye(n*r); % hessian on second order derivative on B
hbc = zeros(n*r); % hessian on partial derivative on B then C
hcc = mu * eye(n*r); % hessian on second order derivative on C
hcb = zeros(n*r); % hessian on partial derivative on C then B

for i = 1:1:n
    for j = 1:1:n
        ei = zeros(n,1);
        ei(i) = 1;
        ej = zeros(n,1);
        ej(j) = 1;
        temp_bc = zeros(n*r);
        temp_bc((i-1)*r+1:i*r, (j-1)*r+1:j*r) = eye(r);
        temp_cb = zeros(n*r);
        temp_cb((i-1)*r+1:i*r, (j-1)*r+1:j*r) = eye(r);
        hbb = hbb + 2*G(i,j)*(reshape(-ei*C(:,j)', [n*r 1]) ... 
              * reshape(-ei*C(:,j)', [n*r 1])');
        hcc = hcc + 2*G(i,j)*(reshape(-B(i,:)'*ej', [n*r 1]) ... 
              * reshape(-B(i,:)'*ej', [n*r 1])');
        hbc = hbc + 2*G(i,j)*(A(i,j) - B(i,:)*C(:,j))*(-temp_bc) + ...
              2*G(i,j)*(reshape(-ei*C(:,j)', [n*r 1]) ... 
              * reshape(-B(i,:)'*ej', [n*r 1])');
        hcb = hcb + 2*G(i,j)*(A(i,j) - B(i,:)*C(:,j))*(-temp_cb) + ...
              2*G(i,j)*(reshape(-B(i,:)'*ej', [n*r 1]) ... 
              * reshape(-ei*C(:,j)', [n*r 1])');
    end
end
h = [hbb, hbc; hcc, hcb];
end

