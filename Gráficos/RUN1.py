import numpy as np
import matplotlib.pyplot as plt
from FuncaoH import FuncaoDeTransferencia

x = np.linspace(-35, 4, 150)
y = np.linspace(-6, 6, 150)
X, Y = np.meshgrid(x, y)
S = X + 1j * Y

coefX = np.array([1, 10, 34, 0])
coefY = np.array([0, 0, 6, 204])

H = FuncaoDeTransferencia(coefY, coefX, S)
print(H)
realPolo, imagPolo = H.poloszeros(coefX)
realZero, imagZero = H.poloszeros(coefY)

modulo = H.magnitude()

# Gráfico dos Polos e Zeros
figura = plt.figure(figsize=(12, 7))
# GRÁFICO DE POLOS E ZEROS
eixo1 = figura.add_subplot(1, 2, 1)
plt.scatter(realPolo, imagPolo, color='blue', marker='x', s=100)
plt.scatter(realZero, imagZero, color='blue', marker='o', s=100)

# CONFIGURAÇÃO DO GRÁFICO DE POLOS E ZEROS

eixo1.axhline(0, color='black', linewidth=0.8)
eixo1.axvline(0, color='black', linewidth=0.8)
eixo1.set_xlabel("Parte Real")
eixo1.set_ylabel("Parte Imaginária")
eixo1.set_title("Números Complexos no Plano")
plt.grid(True)

eixo2 = figura.add_subplot(1, 2, 2, projection='3d')
# CONFIGURAÇÃO IMPRESSÃO DA SUPERFICÍE
eixo2.plot_surface(X, Y, modulo, cmap='viridis', linewidth=0, antialiased=False)
# LEGENDA DO GRÁFICO 3D
eixo2.set_xlabel(r"$\sigma$")
eixo2.set_ylabel(r"$\omega$")
eixo2.set_zlabel('Magnitude |H(s)|')
eixo2.set_title(r"Magnitude de $H(s) = \frac{2s^2+5}{s^2+3s+2}$")
plt.show()
