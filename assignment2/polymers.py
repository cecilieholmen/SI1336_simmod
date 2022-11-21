# %% Import
import random 
import matplotlib.pyplot as plt
import numpy as np


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
        elif random_number == 2:
            x.append(x[-1])
            y.append(y[-1] + 1)
        elif random_number == 3:
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
        elif r_j == 2:
            x.append(x[-1])
            y.append(y[-1] + 1)
        elif r_j == 3:
            x.append(x[-1])
            y.append(y[-1] - 1)
    return [x, y]

def distance_r(walk):
    x = walk[0]
    y = walk[1]
    return np.sqrt(x[-1] ** 2 + y[-1] ** 2)

def self_avoiding_random_walk(x_0, y_0, steps):
    random_walk_dictionary = {(x_0, y_0): 1}
    x = [x_0]
    y = [y_0]
    for i in range(steps):
        random_number = int(4 * random.random())
        if random_number == 0:
            x.append(x[-1] + 1)
            y.append(y[-1])
        elif random_number == 1:
            x.append(x[-1] - 1)
            y.append(y[-1])
        elif random_number == 2:
            x.append(x[-1])
            y.append(y[-1] + 1)
        elif random_number == 3:
            x.append(x[-1])
            y.append(y[-1] - 1)

        if (x[-1], y[-1]) in random_walk_dictionary:
            return False
        else:
            random_walk_dictionary[(x[-1], y[-1])] = 1
    return True

def self_avoiding_random_walk_three_directions(x_0, y_0, steps):
    random_walk_dictionary = {(x_0, y_0): 1}
    x = [x_0]
    y = [y_0]
    for i in range(steps):
        random_number = int(4 * random.random())
        if random_number == 0:
            if len(x) == 1:
                x.append(x[-1] + 1)
                y.append(y[-1])
            elif len(x) > 1 and x[-1] != x[-2] + 1:
                if (x[-1] + 1, y[-1]) in random_walk_dictionary:
                    return False
                else:
                    x.append(x[-1] + 1)
                    y.append(y[-1])
        elif random_number == 1:
            if len(x) == 1:
                x.append(x[-1] - 1)
                y.append(y[-1])
            elif len(x) > 1 and x[-1] != x[-2] - 1:
                if (x[-1] - 1, y[-1]) in random_walk_dictionary:
                    return False
                else:
                    x.append(x[-1] - 1)
                    y.append(y[-1])
        elif random_number == 2:
            if len(x) == 1:
                x.append(x[-1])
                y.append(y[-1] + 1)
            elif len(x) > 1 and y[-1] != y[-2] + 1:
                if (x[-1], y[-1] + 1) in random_walk_dictionary:
                    return False
                else:
                    x.append(x[-1])
                    y.append(y[-1] + 1)
        elif random_number == 3:
            if len(x) == 1:
                x.append(x[-1])
                y.append(y[-1] - 1)
            elif len(x) > 1 and y[-1] != y[-2] - 1:
                if (x[-1], y[-1] - 1) in random_walk_dictionary:
                    return False
                else:
                    x.append(x[-1])
                    y.append(y[-1] - 1)
        else:
            random_walk_dictionary[(x[-1], y[-1])] = 1
    return True

def plot_random_walk_2d(walk, steps):
    plt.clf()
    plt.title(f'2D Random Walk, {steps} steps', fontdict = {'fontsize' : 20})
    plt.plot(walk[0], walk[1], "-x", label="Position")
    plt.xlabel("x-axis", fontsize = 15)
    plt.ylabel("y-axis", fontsize = 15)
    plt.legend(fontsize = 10)
    plt.savefig(f"Random_Walk_{steps}Steps.png")
    plt.axis('square')
    plt.grid()
    plt.show()

def plot_random_walk_generator(walk, steps, m, r0, a, c):
    plt.clf()
    plt.title(f'Random Walk, {steps} steps, m = {m}, r0 = {r0}, a = {a}, c = {c}', fontdict = {'fontsize' : 10})
    plt.plot(walk[0], walk[1], "-x", label="Position")
    plt.xlabel("x-axis", fontsize = 15)
    plt.ylabel("y-axis", fontsize = 15)
    plt.legend(fontsize = 10)
    plt.savefig(f"RWG_ex21b_{m, r0, a, c}.png")
    plt.axis('square')
    plt.grid()
    plt.show()

# %% Exercise 2.1 a)
# Program that generates a two-dimensional random walk with single steps along x or y. 
# The program generates random numbers with random.random().
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
    r0_2 = 2
    a = 3
    a_2 = 5
    c = 4
    c_2 = 6
    m1 = 128
    m2 = 129
    random_walk1 = random_walk_generator(initial_position[0], initial_position[1], r0, a, c, m1, steps)
    plot_random_walk_generator(random_walk1, steps, m1, r0, a, c)

    random_walk2 = random_walk_generator(initial_position[0], initial_position[1], r0, a, c, m2, steps)
    plot_random_walk_generator(random_walk2, steps, m2, r0, a, c)

    random_walk3 = random_walk_generator(initial_position[0], initial_position[1], r0_2, a, c, m1, steps)
    plot_random_walk_generator(random_walk3, steps, m1, r0_2, a, c)

    random_walk4 = random_walk_generator(initial_position[0], initial_position[1], r0_2, a, c, m2, steps)
    plot_random_walk_generator(random_walk4, steps, m2, r0_2, a, c)

    random_walk5 = random_walk_generator(initial_position[0], initial_position[1], r0, a_2, c, m1, steps)
    plot_random_walk_generator(random_walk5, steps, m1, r0, a_2, c)

    random_walk6 = random_walk_generator(initial_position[0], initial_position[1], r0, a_2, c, m2, steps)
    plot_random_walk_generator(random_walk6, steps, m2, r0, a_2, c)

    random_walk7 = random_walk_generator(initial_position[0], initial_position[1], r0, a, c_2, m1, steps)
    plot_random_walk_generator(random_walk7, steps, m1, r0, a, c_2)

    random_walk8 = random_walk_generator(initial_position[0], initial_position[1], r0, a, c_2, m2, steps)
    plot_random_walk_generator(random_walk8, steps, m2, r0, a, c_2)

if __name__ == '__main__':
    exercise_21b()

# %% Exercise 2.1 c)
# Program that generates a two-dimensional random walk with single steps along x or y. 
# The program generates random numbers with random.random().
# The program also calculates the mean square end-to-end distance and plots it as a function of the number of steps.

def exercise_21c():
    initial_position = [0, 0]
    steps_list = [steps for steps in range(10, 1000, 100)]
    number_walks = 1000
    rmsd_list = []
    rmsf_list = []
    sd_list = []

    for steps in steps_list:
        r2_list = []
        r_list = []

        for i in range(number_walks):
            random_walk = random_walk_2d(initial_position[0], initial_position[1], steps)
            r2_list.append(distance_r(random_walk) ** 2)
            r_list.append(distance_r(random_walk))

        variance = np.sum(r2_list) / number_walks - (np.sum(r_list) / number_walks) ** 2
        print(f"Variance: {variance}")

        rmsd_list.append(np.sqrt(np.sum(r2_list) / number_walks))
        rmsf_list.append(np.sqrt(variance * number_walks / (number_walks - 1)))
        sd_list.append(np.sqrt(variance / (number_walks - 1)))

    plt.clf()
    plt.title(f'Random Walk, {number_walks} walks', fontdict = {'fontsize' : 10})
    plt.plot(steps_list, rmsd_list, "-x", label="Root Mean Square Distance")
    plt.plot(steps_list, rmsf_list, "-x", label="Root Mean Square Fluctuation")
    plt.plot(steps_list, sd_list, "-x", label="Standard Deviation")
    plt.xlabel("Number of steps", fontsize = 15)
    plt.ylabel("Value", fontsize = 15)
    plt.legend(fontsize = 10)
    plt.savefig(f"Random_Walk_RMSD.png")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    exercise_21c()
    
# %% Exercise 2.1 d)
# Program that generates a two-dimensional random walk with single steps along x or y that does not cross itself.
# Generate self-avoiding random walks by storing all previously visited sites of the same walk and terminate and discard the walk when a previously visited is revisited.
# How does the fraction of successful walks depend on N? What is the maximum value of N that you can reasonably consider? 
# You can improve the algorithm somewhat by only generating moves in three directions, not back in the direction where you just came from. How does that improve the success?

def exercise_21d():
    initial_position = [0, 0]
    success = 0
    number_walks = 10000
    success_list = []
    success_list_improved = []
    steps_list = [steps for steps in range(1, 100)]
    for steps in steps_list:
        success = 0
        success_improved = 0
        for i in range(number_walks):
            if self_avoiding_random_walk(initial_position[0], initial_position[1], steps) == True:
                success += 1
            if self_avoiding_random_walk_three_directions(initial_position[0], initial_position[1], steps) == True:
                success_improved += 1
        success_list.append(success)
        success_list_improved.append(success_improved)
    
    plt.clf()
    plt.title(f'Self Avoiding Random Walk, {number_walks} walks', fontdict = {'fontsize' : 15})
    plt.plot(steps_list, success_list, "-x", label="Successful walks")
    plt.xlabel("Number of steps, N", fontsize = 15)
    plt.ylabel(f"Number of successfull walks", fontsize = 15)
    plt.legend(fontsize = 10)
    plt.savefig(f"Random_Walk_Self_Avoiding.png")
    plt.grid()
    plt.show()

    accept_list = []
    for i in range(len(success_list)):
        if success_list[i] > 0.25 * success_list[0]:
            accept_list.append(steps_list[i])
    print(f"Maximum acceptable number of steps: {max(accept_list)}")

    plt.clf()
    plt.title(f'Self Avoiding Random Walk Improved, {number_walks} walks', fontdict = {'fontsize' : 15})
    plt.plot(steps_list, success_list_improved, "-x", label="Successful walks, improved algorithm")
    plt.xlabel("Number of steps, N", fontsize = 15)
    plt.ylabel(f"Number of successfull walks", fontsize = 15)
    plt.legend(fontsize = 10)
    plt.grid()
    plt.show()

    plt.clf()
    plt.title(f'Self Avoiding Random Walk, {number_walks} walks', fontdict = {'fontsize' : 15})
    plt.loglog(steps_list, success_list_improved, "-x", label="Successful walks, improved algorithm")
    plt.loglog(steps_list, success_list, "-x", label="Successful walks")
    plt.xlabel("Number of steps, N", fontsize = 15)
    plt.ylabel(f"Number of successfull walks", fontsize = 15)
    plt.legend(fontsize = 10)
    plt.savefig(f"Random_Walk_Self_Avoiding_Improved.png")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    exercise_21d()
# %%
