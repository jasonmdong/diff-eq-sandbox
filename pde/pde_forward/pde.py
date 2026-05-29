import matplotlib.pyplot as plt
import math
import numpy as np

# u_x + u_y + u = e^{x + 2y}, u(x, 0) = 0

h = 0.01
mmax = 2
iterations = (int) (mmax / h)

u = [[0 for i in range(iterations + 1)] for j in range(iterations + 1)]

for j in range(0, iterations):
    y_val = j * h
    for i in range(0, iterations):
        x_val = i * h
        # Forward blows up so using backward instead
        u[i][j + 1] = (2 - h) * u[i][j] - u[i + 1][j] + h * math.exp(x_val + 2 * y_val)
        # u[i][j + 1] = u[i - 1][j] + h * math.exp(x_val + 2 * y_val) - h * u[i][j]

exact = math.exp(mmax + 2 * mmax) / 4 - math.exp(mmax - 2 * mmax) / 4
print(exact)
print(u[iterations - 1][iterations - 1])

from mpl_toolkits.mplot3d import Axes3D

X, Y = np.meshgrid(np.arange(0, mmax + h, h), np.arange(0, mmax + h, h))
U = np.array(u).T

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
plt.savefig('pde_solution.png')

for j in range(0, iterations + 1):
    for i in range(0, iterations + 1):
        if j > i:
            U[j][i] = np.nan
        else:
            U[j][i] -= math.exp(i * h + 2 * j * h) / 4 - math.exp(i * h - 2 * j * h) / 4

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
plt.savefig('pde_diff_solution.png')