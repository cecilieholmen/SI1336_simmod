# %% Import
import random 
import matplotlib.pyplot as plt


# %% Exercise 2.1
# Write a program that generates a two-dimensional random walk with single steps along x or y. 
# The simplest way to do this is by generating a number randomly taken out of the sequence (0,1,2,3) by 
# multiplying random.rnd()by 4 and rounding the result down to an integer using int(). 
# Then you can increase x by one for 0, decrease x by one for 1, and the same for y with 2 and 3. 
# Plot random walks for 10, 100 and 1000 steps.



def random_walk_2d(x_0, y_0, steps):
    x = [x_0]
    y = [y_0]
    for i in range(1, steps + 1):
        random_number = int(4 * random.random())
        if random_number == 0:
            x.append(x[-1] + 1)
            y.append(y[-1])
        elif random_number == 1:
            x.append(x[-1] - 1)
            y.append(y[-1])
        elif random_number == 3:
            x.append(x[-1])
            y.append(y[-1] + 1)
        elif random_number == 4:
            x.append(x[-1])
            y.append(x[-1] - 1)
    return [x, y]

def plot_random_walk_2d(walk, steps):
    plt.clf()
    plt.title(f'2D Random Walk, {steps} steps', fontdict = {'fontsize' : 20})
    plt.plot(walk[0], walk[1], "-x", label="Position")
    plt.xlabel("x-axis", fontsize = 15)
    plt.ylabel("y-axis", fontsize = 15)
    plt.legend(fontsize = 10)
    plt.savefig(f"Random_Walk_{steps}Steps.png")
    plt.grid()
    plt.show()

def exercise_21a():
    initial_position = [0, 0]
    steps_1 = 10
    steps_2 = 100
    steps_3 = 1000
    random_walk_1 = random_walk_2d(initial_position[0], initial_position[1], steps_1)
    plot_random_walk_2d(random_walk_1, steps_1)

    random_walk_2 = random_walk_2d(initial_position[0], initial_position[1], steps_2)
    plot_random_walk_2d(random_walk_2, steps_2)

    random_walk_3 = random_walk_2d(initial_position[0], initial_position[1], steps_3)
    plot_random_walk_2d(random_walk_3, steps_3)

if __name__ == '__main__':
    exercise_21a()


# %%
