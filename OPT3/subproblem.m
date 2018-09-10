function[pkc,d_mk] = subproblem(trust_radius, g, B)
ng = norm(g);
pks = - trust_radius/ng*g;
tmp = g'*B*g;
if  tmp<=0 
    t = 1;
else
    t = min(1, ng^3/trust_radius/tmp);
end
pkc = t*pks;
d_mk = - g'*pkc - .5*pkc'*B*pkc;

end