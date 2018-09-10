function [ g ] = firstOrder( A, G, B, C, mu, n, r )
% This function calculates the first order gradient of the objective
% function.
gb = mu * B; % gradient on B
gc = mu * C; % gradient on C

for i=1:1:n
    for j=1:1:n
        ei = zeros(n,1);
        ei(i) = 1;
        ej = zeros(n,1);
        ej(j) = 1;
        gb = gb + 2*G(i,j)*(A(i,j) - B(i,:)*C(:,j))*(-ei*C(:,j)');
        gc = gc + 2*G(i,j)*(A(i,j) - B(i,:)*C(:,j))*(-B(i,:)'*ej');
    end
end
gbv = reshape(gb, [n*r 1]); % convert B to a vector
gcv = reshape(gc, [n*r 1]); % convert C to a vector
g = [gbv;gcv];
end

