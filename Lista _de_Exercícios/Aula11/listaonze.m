pkg load control
% Taxa de amostragem
fs = 100;
Ts = 1/fs;

% Tempo total e quantidade de elementos
tempo = 10;

t = 0:Ts:tempo;
%t = linspace(0, tempo, N);
N = length(t);

% Função
e = exp(-2*t);
g = 10/13 - (10/13)*cos(3*t).*e - (20/117)*sin(3*t).*e;

% Parâmetros
b = 4;
k = 13;
M = 1;

%   Matrizes
A = [0 1; -(k/M) -(b/M)];
B = [0 1/M]';
C = [1 0];
D = 0;

% Condições iniciais
x0=[0 0]';
u = 10*ones(1, N);

%   PULSOS
##u = zeros(1, N);
##amplitude = 5;
##freq = 5;
##reference = sin(freq*t);
##for i = 1:N
##  if (reference(i) > 0)
##    u(i) = amplitude;
##  else
##    u(i) = -amplitude;
##  endif
##endfor

%   PULSO
##u = zeros(1, N);
##amplitude = 5;
##for i = 1:N
##  if (t(i)<4)
##    u(i) = 5;
##  endif
##endfor


% Modelo Espaço-Estado
sys = ss(A, B, C, D);
[y, t, x] = lsim(sys, u, t, x0);

size(y)

% Gráficos
figure(1);
plot(t, g)
title("Resposta ao impulso a partir da função no tempo");
grid on;

figure(2);
plot(t, y);
title("Resposta ao impulso");
grid on;

##figure(3);
##plot(t, u)
##title("Pulso");
##grid on;

