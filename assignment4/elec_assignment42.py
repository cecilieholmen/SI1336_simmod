# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# %% Global variables

L = 10

# The grid spacing is L/n
delta_x = 0.5
delta_y = delta_x

# The grid is n+1 points along x and y, including boundary points 0 and n
n = int(L/delta_x)

# The number of iterations
nsteps = 1500

# Initialize the grid to 0
v = 10 * 0.9 * np.ones((n+1, n+1))
vnew = np.zeros((n+1, n+1))
v_exact = 10 * np.ones((n+1, n+1)) 

# %% Set the boundary conditions
v[0,:] = 10
v[n,:] = 10
v[:,0] = 10
v[:,n] = 10
#v[n//2, n//2] = 4

# %% Create the plot
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(v, cmap=None, interpolation='nearest')
fig.colorbar(im)

# checker=1: no checkboard, checker=2: checkerboard (note: n should be even)
checker = 2

# perform one step of relaxation
def relax(n, v, checker):
    for check in range(0,checker):
        for x in range(1,n):
            for y in range(1,n):
                if (x*(n+1) + y) % checker == check:
                    v[x,y] = (v[x-1][y] + v[x+1][y] + v[x][y-1] + v[x][y+1])*0.25 #+ 2*np.pi*1

def update(step):
    global n, v, checker

    # FuncAnimation calls update several times with step=0,
    # so we needs to skip the update with step=0 to get
    # the correct number of steps 
    if step > 0:
        relax(n, v, checker)

    # get the max error
    
    err = np.max(np.abs(v - v_exact)/v_exact)
    #err = np.abs(v[5, 5] - v_exact[5, 5])/v_exact[5, 5]
    p = True if err < 0.01 else None
    print(f'Error at step {step}: {err:.3f} {p}')

    im.set_array(v)
    return im,

# we generate nsteps+1 frames, because frame=0 is skipped (see above)
anim = animation.FuncAnimation(fig, update, frames=nsteps+1, interval=200, blit=True, repeat=False)
plt.show()