import numpy as np
import matplotlib.pyplot as plt

# Parameters
K = 10
grid_range = 5
step = 0.1

# Define 2D grid
x = np.arange(-grid_range, grid_range + step, step)
y = np.arange(-grid_range, grid_range + step, step)
X, Y = np.meshgrid(x, y)

# Define centers a_i (3x3 grid)
a_x, a_y = np.meshgrid([-3, 0, 3], [-3, 0, 3])
a_points = np.vstack([a_x.ravel(), a_y.ravel()]).T  # Shape (9, 2)

# Use linspace for ci values: shallowest first, deepest last
c_values = np.linspace(0.5, 0.1, num=len(a_points))  # decreasing ci → increasing depth

# Compute fox-hole function with varying ci
Z = np.zeros_like(X)
for a, c in zip(a_points, c_values):
    dist_sq = (X - a[0])**2 + (Y - a[1])**2
    Z += 1.0 / (dist_sq + c)
Z = K - Z

# Plotting
fig = plt.figure(figsize=(14, 6))

# 3D surface plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='jet', edgecolor='k', linewidth=0.3, antialiased=True)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')
ax1.set_title('Fox-hole Function')
ax1.view_init(elev=20, azim=-45)

# 2D heatmap
ax2 = fig.add_subplot(122)
heatmap = ax2.pcolormesh(X, Y, Z, cmap='jet', shading='auto')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('2D Heatmap')
# fig.colorbar(heatmap, ax=ax2, shrink=0.8)

plt.tight_layout()
plt.show()
