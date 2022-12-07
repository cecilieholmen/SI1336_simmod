# Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

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

# %% Set the boundary conditions
v[0,:] = 10
v[n,:] = 10
v[:,0] = 5
v[:,n] = 5

# %% Create the plot
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(v, cmap=None, interpolation='nearest')
fig.colorbar(im)

# Green's function
G1 = np.zeros((n+1, n+1))
G2 = np.zeros((n+1, n+1))
G3 = np.zeros((n+1, n+1))
G4 = np.zeros((n+1, n+1))

# perform one step of relaxation
def random_walk_of_laplace(n, v, x, y):
    sides = np.zeros(4)
    for i in range(nwalkers):
        x1 = x
        y1 = y
        while True:
            random_number = int(4 * random.random())
            if random_number == 0:
                x1 += 1
            elif random_number == 1:
                x1 -= 1
            elif random_number == 2:
                y1 += 1
            elif random_number == 3:
                y1 -= 1
            if x1 == 0 or x1 == n or y1 == 0 or y1 == n:
                if x1 == 0:
                    sides[0] += 1
                elif x1 == n:
                    sides[1] += 1
                elif y1 == 0:
                    sides[2] += 1
                elif y1 == n:
                    sides[3] += 1
                break
    return sides

def update(step, xy):
    global n, v, G

    x = xy[0]
    y = xy[1]
    g_list = []

    if step > 0:
        random_walk = random_walk_of_laplace(n, v, x, y)
        g_list.append(random_walk)
    
    return g_list 


for x in range(n+1):
    for y in range(n+1):
        xy = [x, y]
        g_list = update(nsteps, xy)
        G1[x, y] = g_list[0][0]
        G2[x, y] = g_list[0][1]
        G3[x, y] = g_list[0][2]
        G4[x, y] = g_list[0][3]

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