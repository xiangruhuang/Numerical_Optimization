function result = derivative(B, C, A, G, mu)
% result is a nxrx2 dimensional gradient vector evaluated at B and C, which can be reshaped to update matrix B and C. 
[n, r] = size(B);
dB = mu*B;
dC = mu*C;

for i = 1:n
    for j = 1:n
        dB(i,:) = dB(i,:) +2*G(i,j)*(B(i,:)*C(:,j)-A(i,j))*C(:,j)';
        dC(:,j) = dC(:,j) +2*G(i,j)*(B(i,:)*C(:,j)-A(i,j))*B(i,:)';
    end
end
result = [reshape(dB, 1, n*r) reshape(dC, 1, n*r)];
end


