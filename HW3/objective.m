function result = objective(B, C, A, G, mu)
% % result is the obj evaluated at B and C
[n, r] = size(B);
result = mu/2*(norm(B,'fro')^2 + norm(C,'fro')^2);

for i = 1:n
    for j = 1:n
        result = result + G(i,j)*(A(i,j)-B(i,:)*C(:,j))^2;
    end
end

end
