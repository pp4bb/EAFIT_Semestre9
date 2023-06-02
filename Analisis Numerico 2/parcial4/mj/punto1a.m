function [K, b, a, u_h, mesh, error_L2, error_Linf] = punto1a(n, u_ex)

% Definir funcion
d1u_ex = diff(u_ex);
d2u_ex = diff(d1u_ex);
f = -d2u_ex;

% Dominio espacial
x_inf = 0;
x_sup = 1;

N = 2*n-1; 
h = 1/n; % tamaño de paso espacial
mesh = x_inf:h/2:x_sup;

% Inicialización de matriz K y vector b
K = zeros(N, N);
b = zeros(N,1);

% Funciones phi y psi
syms x real
syms w z positive integer

phi(x) = piecewise((-1 <= x) & (x <= 0), (1+x)*(1+2*x), (0 < x) & (x <= 1), (1-x)*(1-2*x), abs(x) > 1, 0);
psi(x) = piecewise(abs(x) <= 1/2, 1-4*x.^2, abs(x) > 1/2, 0);

y = (x-w.*(h/2))/(h);

% Función de prueba lambda
lambda(z) = piecewise(mod(z,2) == 0, phi(subs(y,w,z)), mod(z,2) == 1, psi(subs(y,w,z)));

I = 1:N;
V = lambda(I);
D = diff(V);

% Se llena la matriz K
for i = 1:N
    for j=1:N
       K(i,j) = eval(int(D(i)*D(j),x_inf,x_sup));
    end
    b(i) = eval(int(f*V(i),x_inf,x_sup));
end

% Se obtiene el vector de alphas (b)
a = linsolve(K,b);

% Se obtiene la u aproximada
u_h = V*a;

% Se obtiene el error L2
error_L2 = double(int((u_h-u_ex)^2,x,[0,1])^1/2);

% Se obtiene el error L inf
part=1000;
xerror=linspace(x_inf,x_sup,part);
u_hh(x)=u_h;
u_exx(x)=u_ex;
errors=zeros(part,1);
for i=1:part
    errors(i,1)=((double(u_hh(xerror(i))-u_exx(xerror(i))))^2)^1/2;
end
error_Linf = max(errors);

end