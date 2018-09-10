function[BB,CC] = alternating_minimization(G,A,B,C,u)
inds = 1;
tol = 10^(-5);
BB{1} = B;
CC{1} = C;
nmax = 10000;
while inds <= nmax
    BB{inds+1} = next_B(G,A,BB{inds},CC{inds},u);
    CC{inds+1} = next_C(G,A,BB{inds+1},CC{inds},u);
    
    if norm(BB{inds+1}(:) - BB{inds}(:))+norm(CC{inds+1}(:) - CC{inds}(:)) <= tol
            break;
    end
    inds = inds +1 ;
end
f(G,A,BB{end} ,CC{end} ,0)
end
