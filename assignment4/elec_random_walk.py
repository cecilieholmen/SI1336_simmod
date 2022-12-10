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
nwalkers = 1000

# Initialize the grid to 0
v = 10 * 0.9 * np.ones((n+1, n+1))
vnew = np.zeros((n+1, n+1))
v_exact = 10 * np.ones((n+1, n+1)) 

# %% Set the boundary conditions
v[0,:] = 10
v[n,:] = 10
v[:,0] = 5
v[:,n] = 5
#v[n//2, n//2] = 4

# %% Create the plot
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(v, cmap=None, interpolation='nearest')
fig.colorbar(im)

# checker=1: no checkboard, checker=2: checkerboard (note: n should be even)
checker = 1

# perform one step of relaxation
def random_walk_of_laplace(n, v, x, y):
    v_b = np.zeros(nwalkers)
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
                break
        v_b[i] = v[x1, y1]
    v[x, y] = np.mean(v_b)

def update(step):
    global n, v, checker

    x = 4
    y = 4

    # FuncAnimation calls update several times with step=0,
    # so we needs to skip the update with step=0 to get
    # the correct number of steps 

    if step > 0:
        random_walk_of_laplace(n, v, x, y)

    # get the max error
    
    #err = np.max(np.abs(v - v_exact)/v_exact)
    err = np.abs(v[x, y] - v_exact[x, y])/v_exact[x, y]
    p = True if err < 0.01 else None
    print(f'Error at step {step}: {err:.3f} {p}')

    im.set_array(v)
    return im,

# we generate nsteps+1 frames, because frame=0 is skipped (see above)
# anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=200, blit=True, repeat=False)
# plt.show()

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
    v[x, y] = np.mean(v_b)

def update_plot_v(step, walks, xy):
    global n, v

    x = xy[0]
    y = xy[1]
    v_xy_list = []

    # FuncAnimation calls update several times with step=0,
    # so we needs to skip the update with step=0 to get
    # the correct number of steps 
    if step > 0:
        v_xy_list.append(random_walk_of_laplace_more_walks(n, walks, v, x, y))

    print(v_xy_list)

    # get the max error
    err = np.abs(v[x, y] - v_exact[x, y])/v_exact[x, y]
    p = True if err < 0.01 else None
    print(f'Error at step {step}: {err:.3f} {p}')
    
    return v_xy_list

nwalks_list = [walks for walks in range(100, nwalkers, 100)]
v_list = []
xy_list = [[1, 4], [7, 4], [4, 8]]

for i in range(len(xy_list)):
    xy = xy_list[i]
    v_list_xy = []
    for walks in nwalks_list:
        v_list_xy.append(update_plot_v(nsteps, walks, xy))  
    v_list.append(max(v_list_xy))

print(v_list)
print(nwalks_list)

plt.figure()
plt.plot(nwalks_list, v_list, '-x')
plt.xlabel('Number of walks')
plt.ylabel('Potential')
plt.legend(['(1, 4)', '(7, 4)', '(4, 8)'])
plt.grid()
plt.show()
