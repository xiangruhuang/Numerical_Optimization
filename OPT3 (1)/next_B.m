function[outB] = next_B(G,A,B,C,mu)
%% finding optimal B having A,C,G,u via solving linear equations
[n,r ]= size(B);
Q = zeros(n*r);
b = zeros(n*r,1);
for u = 1:n
    for v = 1:r
        for j = 1:n
            for k = 1:r
                Q((u-1)*r+v,u + (k-1)*n) = Q((u-1)*r+v,u + (k-1)*n) + 2*G(u,j)*C(k,j)*C(v,j);
            end
            b((u-1)*r+v,1) = b((u-1)*r+v,1)+2*G(u,j)*A(u,j)*C(v,j);
        end
         Q((u-1)*r+v,u + (v-1)*n) = Q((u-1)*r+v,u + (v-1)*n) + mu;
    end
end
outB = reshape(pinv(Q)*b,n,r);



end
