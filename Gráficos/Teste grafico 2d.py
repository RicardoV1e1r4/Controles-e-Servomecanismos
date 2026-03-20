import numpy as np
import matplotlib.pyplot as plt

# Criando alguns números complexos
zeros = np.array([np.sqrt(3) + 1j, np.sqrt(3) - 1j])
polos = np.array([(np.sqrt(2))/4 + 0.25j*(np.sqrt(2)), np.sqrt(2)/4 - 0.25j*(np.sqrt(2))])
# Parte real e imaginária
realZeros = np.real(zeros)
imagZeros = np.imag(zeros)

realpolos = np.real(polos)
imagpolos = np.imag(polos)

# print(realZeros)
# print(imagZeros)
#
# print(realpolos)
# print(imagpolos)

# Imprimindo o gráfico
plt.figure(figsize=(6, 6))
plt.scatter(realZeros, imagZeros, color='blue', marker='o', s=100)
plt.scatter(realpolos, imagpolos, color='blue', marker='x', s=100)

# Configurações
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel("Parte Real")
plt.ylabel("Parte Imaginária")
plt.title("Números Complexos no Plano")
plt.grid(True)
plt.show()
