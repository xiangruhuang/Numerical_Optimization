function [B,C] = recover_mat(v, n, r)

% recover b first
B = zeros(n, r);
C = zeros(r, n);
for i = 1:n
    start = (i - 1) * r + 1;
    stop = i * r; % start position and end position of bi in v
    B(i, :) = v(start:stop);
end

% then recover C
for i = 1:n
    start = n * r + (i - 1) * r + 1;
    stop = n * r + i * r;
    C(:,i) = v(start:stop);
end

end