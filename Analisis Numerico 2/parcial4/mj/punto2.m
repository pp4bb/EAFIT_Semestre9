function [K, M, P, Q, b, d, a, c, u_h, mesh, error_L2, error_Linf] = punto2(n, u_ex)

% Definir funcion
d1u_ex = diff(u_ex);
d2u_ex = diff(d1u_ex);
d3u_ex = diff(d2u_ex);
f = d3u_ex;

x_inf = 0;
x_sup = 1;
N = n-1; 
h = 1/n;
mesh = x_inf:h:x_sup;

% Sistema matricial #1
M =  zeros(N, N);
K = zeros(N, N);
b = zeros(N,1);

% Sistema matricial #2
P =  zeros(N, N);
Q = zeros(N, N);
d = zeros(N,1);

% Funciones phi y psi
syms x real
syms w z positive integer
phi_j(x) = piecewise((-1 <= x) & (x <= 0), (1+x)^2*(1-2*x), (0 < x) & (x <= 1), (1-x)^2*(1+2*x), abs(x) > 1, 0);
psi_j(x) = piecewise((-1 <= x) & (x <= 0), x*(1+x)^2, (0 < x) & (x <= 1), x*(1-x)^2, abs(x) > 1, 0);

y = (x-w.*h)/(h);

phi(z) =  phi_j(subs(y,w,z));
psi(z) =  psi_j(subs(y,w,z));


I = 1:N;

% Primeras y segundas derivadas de phi y psi
phii = phi(I);
d1phii = diff(phii);
d2phii = diff(d1phii);

psii = psi(I);
d1psii = diff(psii);
d2psii = diff(d1psii);


% Se llenan las matrices
for i = 1:N
    for j=1:N
       K(i,j) = eval(int(d2phii(j)*d1phii(i),x_inf,x_sup));
       M(i,j) = eval(int(d2psii(j)*d1phii(i),x_inf,x_sup));
       P(i,j) = eval(int(d2phii(j)*d1psii(i),x_inf,x_sup));
       Q(i,j) = eval(int(d2psii(j)*d1psii(i),x_inf,x_sup));

    end
    b(i) = eval(int(-f*phii(i),x_inf,x_sup));
    d(i) = eval(int(-f*psii(i),x_inf,x_sup));
end

% Se obtienen los vectores a y c de coeficientes
sol = linsolve([K,M;P,Q],[b;d]);
[nr,~] = size(sol);
a = sol(1:nr/2,:);
c = sol(nr/2 +1 :end ,:);

% Se obtiene la u aproximada
u_h = phii*a + psii*c;

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