function [ite, x, f_new] = alter_direction(n, r, G, A, mu, f, gradient)

x = rand(2 * n * r, 1); % vector of B and C
ite = 0;

while(1)
    % first update bi
    for i = 1:n
        start = (i - 1) * r + 1;
        stop = i * r; % start position and end position of bi in v
        B = mu * eye(r);
        b = 0;
        for j = 1:n
            cj = x((n * r + (j - 1) * r + 1):(n * r + j * r));
            B = B + 2 * G(i,j) * (cj * cj');
            b = b + 2 * G(i,j) * A(i,j) * cj;
        end
        x(start:stop) = B\b;
    end
    
    % then update ci
    for i = 1:n
        B = mu * eye(r);
        b = 0;
        start = n * r + (i - 1) * r + 1;
        stop = n * r + i * r;
        for j = 1:n
            bi = x(((j - 1) * r + 1):(j * r));
            B = B + 2 * G(j,i) * (bi * bi');
            b = b + 2 * G(j,i) * A(j,i) * bi;
        end
        x(start:stop) = B\b;
    end
    ite = ite + 1;
    f_new = f(x, G, A, r, mu);
    if(gradient(x, mu, G, r, A) < 1e-4)
        break
    end
end

    
        