clear;
load('hw4_data.mat');
Atmp = cell2mat(A);
btmp = cell2mat(b);
lr = 0.001;
n = length(A);
[d1,d2]= size(A{1});
c = 0.0001;
for i = 1:length(A)
    AA(i,:,:) = Atmp( (i-1)*size(A{1},1)+1:(i)*size(A{1},1),:);
    bb(i,:) = btmp( (i-1)*size(A{1},1)+1:(i)*size(A{1}));
end
A = AA;
b = bb;
x = rand(n,d2);
max_iter = 100000;
X{1} = x;
iter =1;
tol = 0.0001;
rho = 0.2;
while iter < max_iter
    alphak = 0.1;
    fk = f(A,b,X{end});
    gk = gradient(A,b, X{end});
    xtmp = X{end} - alphak*gk;
    fk1 = f(A,b,xtmp);
    while fk1 > fk + c*alphak*(sum(sum(gk.^2)))
      alphak = alphak*rho;
      xtmp = X{end} - alphak*gk;
     fk1 = f(A,b,xtmp);
    end
    X{end+1} = proximal(xtmp);
    %X{end+1} = xtmp;
    tmp  = X{end} - X{end-1};
    e = sum(abs(tmp(:)))
    if e< tol
        break;
    end
end

