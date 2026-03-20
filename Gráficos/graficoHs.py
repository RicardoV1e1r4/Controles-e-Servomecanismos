import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Define the complex function
def funcaotranferencia(s):
    Xs = s ** 2 + 3 * s + 2
    Ys = 2 * s ** 2 + 5
    return Ys / Xs


# Create a grid for the complex plane
x = np.linspace(-6, 3, 150)
y = np.linspace(-4, 4, 150)
X, Y = np.meshgrid(x, y)
S = X + 1j * Y

# Calculate the magnitude of the output
magnitude = np.abs(funcaotranferencia(S))

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, magnitude, cmap='viridis', linewidth=0, antialiased=False)

# Label the axes
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel(r"$\omega$")
ax.set_zlabel('Magnitude |H(s)|')
ax.set_title(r"Magnitude plot of $H(s) = \frac{2s^2+5}{s^2+3s+2}$")
#\frac{6(s + 34)}{s(s^2 + 10s + 34)}
plt.show()
