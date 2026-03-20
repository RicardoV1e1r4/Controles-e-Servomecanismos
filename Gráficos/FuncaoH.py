import numpy as np
import matplotlib.pyplot as plt


class FuncaoDeTransferencia:
    """
        Classe para representar e analisar uma Função de Transferência polinomial no domínio de Laplace (s).

        Exemplos:
        ----------
        >>> H = FuncaoDeTransferencia([1, 2, 3, 4], [1, 2, 3, 4], 'complex array')
    """

    def __init__(self, numerador, denominador, variavel):
        """
        Inicializa a função de transferência com os coeficientes do polinômio do denominador.

        Parâmetros:
        -----------
        coeficientes : list ou array-like
            Coeficientes do polinômio (ex: [1, 2, 3, 4] para s³ + 2s² + 3s + 4)
        """
        # self.coef = np.array(coeficientes, dtype=float)
        # self.polos = None
        # self.zeros = None
        self.num = np.array(numerador, dtype=float)
        self.dem = np.array(denominador, dtype=float)
        self.s = np.array(variavel)

    def __funcao(self):
        """
        Retorna o valor da função G(s) = N(s)/D(s).
        """
        return np.polyval(self.num, self.s) / np.polyval(self.dem, self.s)
        # Gs = coef[0] * s ** 3 + coef[1] * s ** 2 + coef[2] * s + coef[3]

    def poloszeros(self, coef):
        real = np.real(np.roots(coef))
        imaginario = np.imag(np.roots(coef))
        return real, imaginario

    def magnitude(self):
        """Retorna a magnitude |H(s)|"""
        hs = self.__funcao()
        # Hs = ys/xs
        return np.abs(hs)

# coefX = np.array([1, 10, 34, 0])
# coefY = np.array([0, 2, 0, 5])
#
# x = np.linspace(-6, 3, 150)
# y = np.linspace(-4, 4, 150)
# X, Y = np.meshgrid(x, y)
# S = X + 1j * Y
#
# # Define the complex function
# Xs = coefX[0]*S**3 + coefX[1]*S**2 + coefX[2]*S + coefX[3]
#
# realZeros = np.real(np.roots(coefX))
# imagZeros = np.imag(np.roots(coefX))
#
# realpolos = np.real(np.roots(coefX))
# imagpolos = np.imag(np.roots(coefX))
#
# print(realZeros)
# print(imagZeros)
#
# # print(np.roots(coefY))
# # print(Xs)
#
# # Imprimindo o gráfico
# plt.figure(figsize=(6, 6))
# plt.scatter(realZeros, imagZeros, color='blue', marker='o', s=100)
# # plt.scatter(realpolos, imagpolos, color='blue', marker='x', s=100)
#
# # Configurações
# plt.axhline(0, color='black', linewidth=0.8)
# plt.axvline(0, color='black', linewidth=0.8)
# plt.xlabel("Parte Real")
# plt.ylabel("Parte Imaginária")
# plt.title("Números Complexos no Plano")
# plt.grid(True)
# plt.show()
