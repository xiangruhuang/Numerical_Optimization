function f = f(v, G, A, r, mu)
% matrix stored in column direction
[n, ~] = size(A);
f = mu / 2 * (v' * v); % the latter part don't include the last part first

for i = 1:n
    bi = v(((i - 1) * r + 1):(i * r));
    for j = 1:n
        cj = v((n * r + (j - 1) * r + 1):(n * r + j * r));
        f = f + G(i,j) * (A(i,j) - bi' * cj) ^ 2;
    end
end

end
