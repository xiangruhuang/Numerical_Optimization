function [ pk, g, b ] = calculatePk( A, G, B, C, mu, n, r, delta_k )
% This function calculates the current pk using Cauchy point method
g = firstOrder( A, G, B, C, mu, n, r);
b = secondOrder( A, G, B, C, mu, n, r );
pk = -delta_k*(g./norm(g)); 
condition = g'*b*g;
% determine tk
if condition <= 0
    tk = 1;
else
    tk = min(1, norm(g)^3/(delta_k*condition));
end
pk = tk*pk;
end

