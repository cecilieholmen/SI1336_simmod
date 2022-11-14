# %% Import
import random 
import matplotlib.pyplot as plt


# %% Random walk and plot

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
            y.append(y[-1] - 1)
    return [x, y]

def random_walk_generator(x_0, y_0, r_0, a, c, m, steps):
    x = [x_0]
    y = [y_0]
    r = [r_0]
    for i in range(steps):
        r_n = ((a * r[-1] + c) % m)
        r.append(r_n)
    for j in range(steps):
        r_j = r[j] // (m // 4)
        if r_j == 0:
            x.append(x[-1] + 1)
            y.append(y[-1])
        elif r_j == 1:
            x.append(x[-1] - 1)
            y.append(y[-1])
        elif r_j == 3:
            x.append(x[-1])
            y.append(y[-1] + 1)
        elif r_j == 4:
            x.append(x[-1])
            y.append(y[-1] - 1)
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

def plot_random_walk_generator(walk, steps, m):
    plt.clf()
    plt.title(f'2D Random Walk, {steps} steps, m = {m}', fontdict = {'fontsize' : 18})
    plt.plot(walk[0], walk[1], "-x", label="Position")
    plt.xlabel("x-axis", fontsize = 15)
    plt.ylabel("y-axis", fontsize = 15)
    plt.legend(fontsize = 10)
    plt.savefig(f"Random_Walk_Generator_{m}.png")
    plt.grid()
    plt.show()

# %% Exercise 2.1 a)
# Program that generates a two-dimensional random walk with single steps along x or y. 
# The program should plot generated random numbers with random.random().
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

# %% Exercise 2.1 b)
# Program that generates a two-dimensional random walk with single steps along x or y. 
# The program generates random numbers with a generator.

def exercise_21b():
    initial_position = [0, 0]
    steps = 100
    r0 = 1
    a = 3
    c = 4
    m = 128
    random_walk = random_walk_generator(initial_position[0], initial_position[1], r0, a, c, m, steps)
    plot_random_walk_generator(random_walk, steps, m)

if __name__ == '__main__':
    exercise_21b()

# %%
