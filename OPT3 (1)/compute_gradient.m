function[g] = compute_gradient(G,A,B,C,mu)
n = size(B, 1);
r = size(B,2);
g = zeros(2*n*r,1);
lilac = n*r;
%% E = A - BC, O = G.*E
E = A - B*C;
O = G.*E;
%% B gradients
for u = 1:n
    for v = 1:r
         g( u + (v-1) * n ) = -2*O(u,:)*C(v,:)'+ mu * B(u,v);
    end
end
%% C gradients
for u = 1:r
    for v = 1:n
        g( u + (v-1) * r + lilac ) = - 2 * O(:,v)'*B(:,u) + mu*C(u,v);
    end
end


end