# %% Imports
import random
import numpy as np
import matplotlib.pyplot as plt

# Global variables
L = 10

# The grid spacing is L/n
delta_x = 1
delta_y = delta_x

# The grid is n+1 points along x and y, including boundary points 0 and n
n = int(L/delta_x)

# The number of iterations
nsteps = 100

# The number of walkers
nwalkers = 10_000

# Initialize the grid to 0
v = np.zeros((n+1, n+1))

# Set the boundary conditions
v[0, :] = 10
v[n, :] = 10
v[:, 0] = 5
v[:, n] = 5

# All on the left side (should give max for 5,3)
# v[3, 0] = 20
# v[7, 0] = 20
# v[4, 0] = 20
# v[5, 0] = 20
# v[6, 0] = 20

# Three on the top, one on the left and one on the right (should give max for 3,5)
# v[3, 0] = 20
# v[3, n] = 20
# v[0, 4] = 20
# v[0, 5] = 20
# v[0, 6] = 20

# %% Create the plot
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(v, cmap=None, interpolation='nearest')
fig.colorbar(im)

# %% Green's function
G1 = np.zeros_like(v)
G2 = np.zeros_like(v)
G3 = np.zeros_like(v)
G4 = np.zeros_like(v)

def random_walk_of_laplace(n, v, x0, y0):
    sides = np.zeros(4)
    for i in range(nwalkers):
        x = x0
        y = y0
        while True:
            random_number = int(4 * random.random())
            if random_number == 0:
                x += 1
            elif random_number == 1:
                x -= 1
            elif random_number == 2:
                y += 1
            elif random_number == 3:
                y -= 1
            if x == 0 or x == n or y == 0 or y == n:
                if x == 0:
                    sides[0] += v[x, y]
                elif x == n:
                    sides[1] += v[x, y]
                elif y == 0:
                    sides[2] += v[x, y]
                elif y == n:
                    sides[3] += v[x, y]
                break
    return sides / nwalkers

for x in range(1, n):
    for y in range(1, n):
        random_walk = random_walk_of_laplace(n, v, x, y)
        G1[x, y] = random_walk[0]
        G2[x, y] = random_walk[1]
        G3[x, y] = random_walk[2]
        G4[x, y] = random_walk[3]

# Calculate v(x,y) from G(x,y) in the site (3, 5)
v[3, 5] = (G1[3, 5] + G2[3, 5] + G3[3, 5] + G4[3, 5])
print(f"v(3, 5) = {v[3, 5]}")

# Calculate v(x,y) from G(x,y) in the site (5, 3)
v[5, 3] = (G1[5, 3] + G2[5, 3] + G3[5, 3] + G4[5, 3])
print(f"v(5, 3) = {v[5, 3]}")

# %% 
# Plot the results
plt.figure()
plt.contourf(G1)
plt.colorbar()
plt.title('G1')
plt.savefig('G1.png')
plt.show()

plt.figure()
plt.contourf(G2)
plt.colorbar()
plt.title('G2')
plt.savefig('G2.png')
plt.show()

plt.figure()
plt.contourf(G3)
plt.colorbar()
plt.title('G3')
plt.savefig('G3.png')
plt.show()

plt.figure()
plt.contourf(G4)
plt.colorbar()
plt.title('G4')
plt.savefig('G4.png')
plt.show()