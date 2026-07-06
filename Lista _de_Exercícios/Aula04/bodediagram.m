pkg load control

s = tf('s')
% Função de Transferência
G=1000/((s^2 + 0.1*s + 10)*(s^2 + 2*s + 500))
bode(G)

x=22.31;
H=abs(1000/(((j*x)^2 + 0.1*(j*x) + 10)*((j*x)^2 + 2*(j*x) + 500)))

d1=abs((j*x)^2 + 0.1*(j*x) + 10)
d2=abs((j*x)^2 + 2*(j*x) + 500)

