function plot_aprox(u_h, mesh, n)

syms x;
plot_name = 'n='+string(n);
u_h(x) = u_h;
plot(mesh, eval(subs(u_h,mesh)), 'DisplayName', plot_name)
hold on
legend()
title('Exacta vs Aproximada')

end