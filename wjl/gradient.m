function[g] = gradient(A,b,x)
%% A is 10*300*1000 b is 10*300 x is 10*1000 g is 10*1000
g = zeros(size(A,1),size(A,3));
for n = 1:size(A,1)
    g(n,:) = 2*(squeeze(A(n,:,:))*squeeze(x(n,:))' - squeeze(b(n,:))')'*squeeze(A(n,:,:));
end
end