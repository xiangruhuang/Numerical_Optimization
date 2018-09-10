function Hessian = Hessian(G, v, r, mu, A)

n = length(v) / (2 * r);
Hessian = zeros(length(v), length(v));
% we first fill in the diagonal matrix
% first fill in bi
for i = 1:n
    start = (i - 1) * r + 1;
    stop = i * r; % start position and end position of bi in v
    Hessian_temp = mu * eye(r) / 2;
    for j = 1:n
        cj = v((n * r + (j - 1) * r + 1):(n * r + j * r));
        g = G(i,j);
        Hessian_temp = Hessian_temp + g * (cj * cj'); % plus half
    end
    Hessian(start:stop, start:stop) = Hessian_temp;
end
% fill in ci then
for i = 1:n
    start = n * r + (i - 1) * r + 1;
    stop = n * r + i * r; 
    Hessian_temp = mu * eye(r) / 2;
    for j = 1:n
        bj = v(((j - 1) * r + 1):(j * r));
        g = G(i,j);
        Hessian_temp = Hessian_temp + g * (bj * bj'); % plus half
    end
    Hessian(start:stop, start:stop) = Hessian_temp;
end
% fill in bicj joint part
for i = 1:n
    start_b = (i - 1) * r + 1;
    stop_b = i * r; % start position and end position of bi in v
    bi = v(start_b:stop_b);
    for j = 1:n
        start_c = n * r + (j - 1) * r + 1;
        stop_c = n * r + j * r; 
        cj = v(start_c:stop_c);
        Hessian(start_b:stop_b, start_c:stop_c) = 2 * G(i,j) * ((cj' * bi - A(i,j)) * eye(r) + bi * cj');
    end
end
% fill in the other part
Hessian = Hessian + Hessian';
end