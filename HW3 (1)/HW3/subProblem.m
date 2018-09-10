function [ mk ] = subProblem( fk, pk, gk, bk )
% This problem calculates the sub problem value.
mk = fk + gk'* pk + 0.5*pk'*bk*pk;

end

