function[g,H] = compute_gradient_Hessian(G,A,B,C,mu)
n = size(B, 1);
r = size(B,2);
g = zeros(2*n*r,1);%gradients
H =  zeros(2*r*n,2*n*r);%hessian
lilac = n*r; %%offset
%% E = A - BC, O = G.*E
E = A - B*C;
O = G.*E;
%% B gradients
for u = 1:n
    for v = 1:r
         g( u + (v-1) * n ) = -2*O(u,:)*C(v,:)'+ mu * B(u,v);
         for k = 1:r
             H(  u + (v-1) * n ,  u + ( k -1) * n ) = H(  u + (v-1) * n ,  u + ( k -1) * n ) +  2 * sum(C(k , : ).* C( v, : ).* G ( u , : ) );
         end
         H(  u + (v-1) * n ,  u + ( v -1) * n ) = H(  u + (v-1) * n ,  u + ( v -1) * n )+ mu;
        for j = 1:n
            for k = 1:r
                H( u + (v-1) * n ,  k + (j-1)*r +lilac)  = H( u + (v-1) * n ,  k + (j-1)*r +lilac)  +  2*G(u,j)*B(u,k)*C(v,j);
            end
             H( u + (v-1) * n ,  v + (j-1)*r +lilac)  = H( u + (v-1) * n ,   v + ( j - 1 )*r +lilac) -  2*O(u,j);
        end
    end
end
%% C gradients
for u = 1:r
    for v = 1:n
        g( u + (v-1) * r + lilac ) = - 2 * O(:,v)'*B(:,u) + mu*C(u,v);
        for k = 1:r
             H(lilac + u + (v-1) * r ,lilac + k + (v-1) * r ) = H(lilac + u + (v-1) * r ,lilac + k + (v-1) * r ) +  2 * sum(B(:,k).*B(:,u).*G(:,v));
        end
       H(lilac + u + (v-1) * r ,lilac + u + (v-1) * r ) = H(lilac + u + (v-1) * r ,lilac + u + (v-1) * r )  + mu;
        for i = 1:n
            for k = 1:r
                H( lilac + u + (v-1) * r ,  i + (k-1)*n ) = H( lilac + u + (v-1) * r ,  i + (k-1)*n ) + 2*C(k,v)*G(i,v)*B(i,u);
            end
            H( lilac + u + (v-1) * r ,  i + (u-1)*n ) = H( lilac + u + (v-1) * r ,  i + (u-1)*n )  - 2*O(i,v);
        end
    end
end


end