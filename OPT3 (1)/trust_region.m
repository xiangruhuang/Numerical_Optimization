function[BB,CC] = trust_region(G,A,B,C,u)
trust_radius = 2;
tr = 1;
yita = 0.2;
inds = 1;
tol = 10^(-5);
n = size(B,1);
r  = size(C,1);
BB{1} = B;
CC{1} = C;
iter = 1;
nmax = 10000;
while iter <= nmax
    [g,H] =  compute_gradient_Hessian(G,A,BB{inds} ,CC{inds} ,u);
    [pk,dmk] = subproblem(tr,g,H);
    Bpk = reshape(pk(1:n*r),n,r);
    Cpk = reshape(pk(n*r+1:end),r,n);
    Btmp = BB{inds} + Bpk;
    Ctmp = CC{inds} + Cpk;
    rho = (f(G,A,BB{inds},CC{inds},u) - f(G,A,Btmp,Ctmp,u))/dmk;
     if rho < 0.25
        tr = 0.25*tr;
    else
        if rho > 0.75 && norm(pk) == tr
            tr = min(trust_radius,2*tr);
        end
    end
    if rho > yita
        BB{inds+1} = Btmp;
        CC{inds+1} = Ctmp;
        if norm(BB{inds+1}(:) - BB{inds}(:))+norm(CC{inds+1}(:) - CC{inds}(:)) <= tol
            break;
        end
        inds = inds+1;
    end
    iter = iter +1;
    f(G,A,BB{end} ,CC{end} ,0)
end

end
