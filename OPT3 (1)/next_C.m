function[outC] = next_C(G,A,B,C,mu)
%% finding optimal C having A,B,G,u via solving linear equations
[n,r ]= size(B);
Q = zeros(n*r);
b = zeros(n*r,1);
for u = 1:r
    for v = 1:n
        for i = 1:n
            for k = 1:r
                Q((u-1)*n+v, k + (v-1)*r) = Q((u-1)*n+v,k + (v-1)*r) + 2*G(i,v)*B(i,k)*B(i,u);
            end
            b((u-1)*n+v,1) = b((u-1)*n+v,1)+2*G(i,v)*A(i,v)*B(i,u);
        end
         Q((u-1)*n+v,u + (v-1)*r) = Q((u-1)*n+v,u + (v-1)*r) + mu;
    end
end
outC = reshape(pinv(Q)*b,r,n);



end
