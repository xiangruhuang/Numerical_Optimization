function [ LHS, RHS ] = fixC( A, G, C, mu, n, r )
% Fix C, calculate the partial derivative of B and set it to zero 
% to get the local minimum.
% we have n*r equations and n*r unknown variables. And we add uB first.
LHS = eye(n*r, n*r)*mu;  
RHS = zeros(n, r); % For each equation, we have one value.

for i=1:1:n
    row_equ = zeros(r, r); % r equations for r varaibles in nth row
    for j=1:1:n
        %save time since the matrix is largely sparsed
        if G(i,j) ~= 0
            row_equ = row_equ + 2*G(i,j)*C(:,j)*C(:,j)';
            ei = zeros(n,1);
            ei(i) = 1;
            RHS = RHS + 2*G(i,j)*A(i,j)*(ei*C(:,j)');
        end
    end
    LHS((i-1)*r+1:i*r,(i-1)*r+1:i*r) = LHS((i-1)*r+1:i*r,(i-1)*r+1:i*r) + row_equ;
end
RHS = reshape(RHS', [n*r 1]);
end

