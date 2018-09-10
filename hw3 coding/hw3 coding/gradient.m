function gradient = gradient(v, mu, G, r, A)

% v = [b1,b2,...,bn, c1,c2,...,cn]
% input target function and current vector v
% gradient is a n * 1 vector
gradient = zeros(size(v));
n = length(v) / (2 * r);
% update bi first
for i = 1:n
    start = (i - 1) * r + 1;
    stop = i * r; % start position and end position of bi in v
    bi = v(start:stop);
    gradient_temp = mu * v(start:stop);
    for j = 1:n
        cj = v((n * r + (j - 1) * r + 1):(n * r + j * r));
        g = G(i, j);
        a = A(i,j);
        gradient_temp = gradient_temp + 2 * g * (cj' * bi - a) * cj;
    end
    gradient(start:stop) = gradient_temp;
end

% update ci then
for i = 1:n
    start = n * r + (i - 1) * r + 1;
    stop = n * r + i * r;
    ci = v(start:stop);
    gradient_temp = mu * v(start:stop);
    for j = 1:n
        bj = v(((j - 1) * r + 1):(j * r));
        g = G(j,i);
        a = A(j,i);
        gradient_temp = gradient_temp + 2 * g * (bj' * ci - a) * bj;
    end
    gradient(start:stop) = gradient_temp;
end

    
        
