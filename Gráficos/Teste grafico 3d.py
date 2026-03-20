import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Criando a malha de pontos no plano complexo
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Criando a variável complexa z
Z = X + 1j*Y

# Função complexa
F = Z**2

# Módulo da função
modulo = np.abs(F)

# Plot 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, modulo, cmap='viridis')

ax.set_xlabel("Parte Real de z")
ax.set_ylabel("Parte Imaginária de z")
ax.set_zlabel("|f(z)|")
ax.set_title("Módulo de f(z) = z²")
plt.show()
