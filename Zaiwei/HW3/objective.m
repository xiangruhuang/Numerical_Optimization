function [ val ] = objective( A, G, B, C, mu, n )
% This function calculates the value of the objective function.
val = mu*sum(diag(B' * B)) + mu*sum(diag(C' * C));
for i=1:1:n
    for j=1:1:n
        val = val + G(i,j)*(A(i,j) - B(i,:)*C(:,j))^2;
    end
end

end

