%% home work 3
n = 10;
r = 8;
%G = randi([0,1],n,n);
%G = ones(10);
%A = rand(n,n);
%B = rand(n,r);
%C = rand(r,n);
u = 0.01;
%%trust_region method
[BB,CC] = trust_region(G,A,B,C,u);
B_trustregion = BB{end};
C_trustregion  = CC{end};
%%alternating_minimization
[BB,CC] = alternating_minimization(G,A,B,C,u);
B_alternating_minimization = BB{end};
C_alternating_minimization = CC{end};

