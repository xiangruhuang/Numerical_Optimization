function[val] = f(A,b,x)
%% A is 10*300*1000 b is 10*300 x is 10*1000 g is 10*1000
val = 0;
for n = 1:size(A,1)
    aaa = (squeeze(A(n,:,:))*squeeze(x(n,:))' - squeeze(b(n,:))');
    val = val + aaa'*aaa;
end


end
