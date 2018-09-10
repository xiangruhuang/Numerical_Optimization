function[out] = f(G,A,B,C,u)
%% compute the value of function f
out1 = G.*(A-B*C).^(2) ;
out2 = B.*B*u;
out3 = C.*C*u;
out=sum(out1(:))+sum(out2(:))+sum(out3(:));
end