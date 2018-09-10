function[f] = compute_value(G,A,B,C,u)
f = G.*((A-B*C).^2) + 0.5*u*(B.^2 + C.^2);
f = f(:);
f = sum(f);



end