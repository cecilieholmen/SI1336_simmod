# %% Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# %% Global variables

L = 10

# The grid spacing is L/n
delta_x = 1
delta_y = delta_x

# The grid is n+1 points along x and y, including boundary points 0 and n
n = int(L/delta_x)

# The number of iterations
nsteps = 100

# The number of walkers
nwalkers = 100_000

# Initialize the grid
v = np.zeros((n+1, n+1))

# %% Set the boundary conditions
v[0,:] = 10
v[n,:] = 10
v[:,0] = 5
v[:,n] = 5

def random_walk_of_laplace_more_walks(n, walks, v, x, y):
    v_b = np.zeros(walks)
    for i in range(walks):
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
                break
        v_b[i] = v[x1, y1]
    return np.mean(v_b)

def update_plot_v(step, walks, xy):
    global n

    x = xy[0]
    y = xy[1]
    v_xy_list = []

    if step > 0:
        v_xy_list.append(random_walk_of_laplace_more_walks(n, walks, v, x, y))
    
    return v_xy_list

nwalks_list = [walks for walks in np.geomspace(100, nwalkers, 20, dtype=int)]
v_list = []
xy_list = [[1, 4], [7, 4], [4, 8]]

for i in range(len(xy_list)):
    xy = xy_list[i]
    v_list_xy = []
    for walks in nwalks_list:
        v_list_xy.append(update_plot_v(nsteps, walks, xy))  
    v_list.append(v_list_xy)

plt.figure()
plt.plot(nwalks_list, v_list[0], '-x', nwalks_list, v_list[1], '-x', nwalks_list, v_list[2], '-x')
plt.xlabel('Number of walks')
plt.ylabel('Potential')
plt.legend(['(1, 4)', '(7, 4)', '(4, 8)'])
plt.grid()
plt.savefig('elec_random_walk_2.png')
plt.show()

# %%
