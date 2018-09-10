function [ G ] = generateSparseM( grid_size, prob )
%This function generate a sparse observation pattern G, with size:
%grid size x grid size and the probability of generating 1 lower than 30%.

G = zeros(grid_size);

for i=1:1:grid_size
    for j=1:1:grid_size
        if rand < prob
            G(i,j) = 1;
        end
    end
end

end

