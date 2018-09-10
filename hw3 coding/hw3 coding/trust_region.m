function [ite, x, f_new] = trust_region(n, r, G, A, mu, gradient, Hessian, f)

% characterizing B and C into vectors and combine them together
Delta = 2; delta = Delta / 2; eta = 0.2; x = rand(2 * n * r, 1); % vector of B and C
ite = 0;

while(1)
    %obtain pk
    gk = gradient(x, mu, G, r, A);
    % cauchy point method
    pk = -delta * (gk / norm(gk));
    Bk = Hessian(G, x, r, mu, A);
    thres = gk' * Bk * gk;
    if (thres <= 0)
        tauk = 1;
    else
        tauk = min(1, norm(gk)^3 / (delta * thres));
    end
    pk = tauk * pk;
    x_new = x + pk;
    gk_new = gradient(x_new, mu, G, r, A);
    Bk_new = Hessian(G, x_new, r, mu, A);
    f_prev = f(x, G, A, r, mu);
    f_new = f(x_new, G, A, r, mu);
    mk0 = f_prev + gk' * pk + 0.5 * pk' * Bk * pk;
    mk1 = f_new + gk_new' * pk + 0.5 * pk' * Bk_new * pk;
    rho_k = (f_prev - f_new) / (mk0 - mk1);
    % evaluate rho
    if (rho_k < 1/4)
        delta = delta * 0.25;
    else
        if (rho_k > 0.75 && norm(pk) == delta)
            delta = min(2 * delta, Delta);
        end
    end
    
    if (rho_k >= eta)
        x = x_new;
    end
    ite = ite + 1;
    
    if (norm(gk) < 1e-4)
        break
    end
end
