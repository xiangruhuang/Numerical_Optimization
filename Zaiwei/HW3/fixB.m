function [ LHS, RHS ] = fixB( A, G, B, mu, n, r )
% Fix B, calculate the partial derivative of C and set it to zero 
% to get the local minimum.
% we have n*r equations and n*r unknown variables. And we add uC first.
LHS = eye(n*r, n*r)*mu;  
RHS = zeros(r, n); % For each equation, we have one value.

for j=1:1:n
    row_equ = zeros(r, r); % r equations for r varaibles in nth column
    for i=1:1:n
        %save time since the matrix is largely sparsed
        if G(i,j) ~= 0
            row_equ = row_equ + 2*G(i,j)*B(i,:)'*B(i,:);
            ej = zeros(n,1);
            ej(j) = 1;
            RHS = RHS + 2*G(i,j)*A(i,j)*(B(i,:)'*ej');
        end
    end
    LHS((j-1)*r+1:j*r,(j-1)*r+1:j*r) = LHS((j-1)*r+1:j*r,(j-1)*r+1:j*r) + row_equ;
end
RHS = reshape(RHS, [r*n 1]);
end

