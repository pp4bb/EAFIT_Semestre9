function [K, b, a, u_h] = punto1a(n)

N = 2*n-1; % este es el n considerando el problema con los nodos medios y los nodos completos
h = 1/n;
x_inf = 0;
x_sup = 1;

% Construyo malla
mesh = x_inf+h/2:h/2:x_sup-h/2;

% Construyo matriz y vectores
K = zeros(N,N);
b = zeros(N,1);

syms x;
% f = -8;
f = 2;
% f = -pi^2*sin(pi*x);
for i=1:N
    lambdai_prima = lambda_j(mesh, i, h, true);
    for j=1:N
        lambdaj_prima = lambda_j(mesh, j, h, true);
        K(i,j) = int(lambdaj_prima*lambdai_prima, x, [x_inf x_sup]);
    end
    v_h = lambda_j(mesh, i, h, false);
    b(i,1) = int(f*v_h, x, [x_inf x_sup]);
end

a = linsolve(K,b);
u_h = 0;
for j=1:N
    lambdaj = lambda_j(mesh, j, h, false);
    u_h = u_h + a(j)*lambdaj;
end

syms x;
% u_ex(x) = 4*x^2-4*x;
% u_ex(x) = sin(pi*x);
new_mesh = x_inf:h/2:x_sup;
u_ex(x) = x-x^2;
fplot(u_ex,[0,1])
hold on
u_h(x) = u_h;
plot(new_mesh, u_h(new_mesh))
legend('u exacta','u_h')

end


%%%%%%%%%%%%%%%%%%%%%%%%%% Función phi %%%%%%%%%%%%%%%%%%%%%%%%%%
function resp = phi_j(mesh, j, h, derivative)

xj = mesh(j);

% Definir funcion phi
syms phi(x)
phi(x) = piecewise((-1 <= x) & (x <= 0), (1+x)*(1+2*x), (0 < x) & (x <= 1), (1-x)*(1-2*x), abs(x) > 1, 0);

phi_j = phi((x-xj)/h);
resp = phi_j;
if derivative == true
    phi_j_prima = diff(phi_j);
    resp = phi_j_prima;
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%% Función psi %%%%%%%%%%%%%%%%%%%%%%%%%%
function resp = psi_j(mesh, j, h, derivative)

xj = mesh(j);

% Definir funcion psi
syms psii(x)
psii(x) = piecewise(abs(x)<= 1/2, (1-4*x^2), abs(x) > 1/2, 0);

psi_j = psii((x-xj)/h);
resp = psi_j;
if derivative == true
    psi_j_prima = diff(psi_j);
    resp = psi_j_prima;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%% Función lambda %%%%%%%%%%%%%%%%%%%%%%%%%%
function resp_funcion = lambda_j(mesh, j, h, derivative)

% if mod(j,2) == 0
%     resp_funcion = psi_j(mesh, j, h, derivative);
% else
%     resp_funcion = phi_j(mesh, j, h, derivative);
% end

if mod(j,2) == 0
    resp_funcion = phi_j(mesh, j, h, derivative);
else
    resp_funcion = psi_j(mesh, j, h, derivative);
end

end