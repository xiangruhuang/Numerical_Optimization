function[u] = proximal(x)
% x 
d = size(x,2);% dimension
n = size(x,1); % #x
u = zeros(n,d);
for i = 1:d
    [xtmp,I] = sort(x(:,i));
    for k = 1: n
        iindex(k) = find(I == k);
    end
    for j = 1:n
        u(j,i) = xtmp(j,1) + (n - 2*j + 1);
    end
    for j = 2:n
        if u(j,i)  < u(j-1,i) 
            index = j-1;
            val =  xtmp(j,1) +  xtmp(j-1,1) ;
            count = 2;
            while true
                if index - 1>0 && u(index,i) ==  u(index-1,i)
                    count = count +1;
                    val = val +  xtmp(index-1,1);
                    index = index - 1;
                else
                    break
                end
            end
            for k = index:j
                u(k,i) = val/count;
            end
        end
    end
    u(:,i) = u(iindex,i);
end
end

      