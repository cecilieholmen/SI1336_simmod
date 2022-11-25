# %% Imports
#!/bin/python3

# Template for traffic simulation
# BH, MP 2021-11-15, latest version 2022-11-1.

"""
    This template is used as backbone for the traffic simulations.
    Its structure resembles the one of the pendulum project, that is you have:
    (a) a class containing the state of the system and it's parameters
    (b) a class storing the observables that you want then to plot
    (c) a class that propagates the state in time (which in this case is discrete), and
    (d) a class that encapsulates the aforementioned ones and performs the actual simulation
    You are asked to implement the propagation rule(s) corresponding to the traffic model(s) of the project.
"""

import math
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy.random as rng
import numpy as np


# %% Class for the state of the system
class Cars:

    """ Class for the state of a number of cars """

    def __init__(self, numCars=5, roadLength=50, v0=1):
        self.numCars    = numCars
        self.roadLength = roadLength
        self.t  = 0
        self.x  = []
        self.v  = []
        self.c  = []
        for i in range(numCars):
            self.x.append(i)        # the position of the cars on the road
            self.v.append(v0)       # the speed of the cars
            self.c.append(i)        # the color of the cars (for drawing)

    def distance(self, i):
        # TODO: Implement the function returning the PERIODIC distance 
        # between car i and the one in front 
        return (self.x[(i+1)%self.numCars] - self.x[i]) % self.roadLength


class Observables:

    """ Class for storing observables """

    def __init__(self):
        self.time = []          # list to store time
        self.flowrate = []      # list to store the flow rate
        

class BasePropagator:

    def __init__(self):
        return
        
    def propagate(self, cars, obs):

        """ Perform a single integration step """
        
        fr = self.timestep(cars, obs)

        # Append observables to their lists
        obs.time.append(cars.t)
        obs.flowrate.append(fr)  # CHANGE!
              
    def timestep(self, cars, obs):
        """ Virtual method: implemented by the child classes """
        return NotImplementedError
      
        
class ConstantPropagator(BasePropagator) :
    
    """ 
        Cars do not interact: each position is just 
        updated using the corresponding velocity 
    """
    
    def timestep(self, cars, obs):
        for i in range(cars.numCars):
            cars.x[i] += cars.v[i]
        cars.t += 1
        return 0

# TODO
# HERE YOU SHOULD IMPLEMENT THE DIFFERENT CAR BEHAVIOR RULES
# Define you own class which inherits from BasePropagator (e.g. MyPropagator(BasePropagator))
# and implement timestep according to the rule described in the project

class MyPropagator(BasePropagator) :

    def __init__(self, vmax, p):
        BasePropagator.__init__(self)
        self.vmax = vmax
        self.p = p

    def timestep(self, cars, obs):
        # TODO Here you should implement the car behaviour rules
        for i in range(cars.numCars):
            if cars.v[i] < self.vmax:
                cars.v[i] += 1
            distance = Cars.distance(cars, i)
            if distance <= cars.v[i]:
                cars.v[i] = distance - 1
            if cars.v[i] > 0:
                if rng.random() < self.p:
                    cars.v[i] -= 1
            cars.x[i] += cars.v[i]

        fr = sum(cars.v) / cars.roadLength
        cars.t += 1
        return fr

############################################################################################

def draw_cars(cars, cars_drawing):

    """ Used later on to generate the animation """
    cars_drawing.clear()
    cars_drawing.axis('off')
    theta = []
    r     = []

    for position in cars.x:
        # Convert to radians for plotting  only (do not use radians for the simulation!)
        theta.append(position * 2 * math.pi / cars.roadLength)
        r.append(1)

    return cars_drawing.scatter(theta, r, c=cars.c, cmap='hsv')


def animate(framenr, cars, obs, propagator, road_drawing, stepsperframe):

    """ Animation function which integrates a few steps and return a drawing """

    for it in range(stepsperframe):
        propagator.propagate(cars, obs)

    return draw_cars(cars, road_drawing),


class Simulation:

    def reset(self, cars=Cars()) :
        self.cars = cars
        self.obs = Observables()

    def __init__(self, cars=Cars()) :
        self.reset(cars)

    def plot_observables(self, title="simulation"):
        plt.clf()
        plt.title(title, fontsize=20)
        plt.plot(self.obs.time, self.obs.flowrate)
        plt.xlabel('time', fontsize=20)
        plt.ylabel('flow rate', fontsize=20)
        plt.savefig(title + ".pdf")
        plt.show()

    # Run without displaying any animation (fast)
    def run(self,
            propagator,
            numsteps=200,           # final time
            title="simulation",     # Name of output file and title shown at the top
            ):

        for it in range(numsteps):
            propagator.propagate(self.cars, self.obs)

        self.plot_observables(title)

    # Run while displaying the animation of bunch of cars going in circe (slow-ish)
    def run_animate(self,
            propagator,
            numsteps=200,           # Final time
            stepsperframe=1,        # How many integration steps between visualising frames
            title="simulation",     # Name of output file and title shown at the top
            ):

        numframes = int(numsteps / stepsperframe)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.axis('off')
        # Call the animator, blit=False means re-draw everything
        anim = animation.FuncAnimation(plt.gcf(), animate,  # init_func=init,
                                       fargs=[self.cars,self.obs,propagator,ax,stepsperframe],
                                       frames=numframes, interval=50, blit=False, repeat=False)
        plt.show()

        # If you experience problems visualizing the animation and/or
        # the following figures comment out the next line 
        plt.waitforbuttonpress(30)

        self.plot_observables(title)

    # Run without displaying any animation (fast)
    def run_ex22a(self,
            propagator,
            numsteps=200,           # final time
            ):

        for it in range(numsteps):
            propagator.propagate(self.cars, self.obs)

        return [self.obs.flowrate, self.obs.time]
    

# %% Exercise 2.2 a)
def exercise22a() :
    numCars = [c for c in range(0, 55, 5)]
    vmax = 2
    p = 0.5
    roadLength = 50
    numsteps = 200
    flowrate_list = []
    density_list = []

    for n in numCars:
        density_list.append(n / roadLength)

        cars = Cars(n, roadLength=roadLength)
        sim = Simulation(cars)
        propagator = MyPropagator(vmax, p)
        flowrate_list.append(np.mean(sim.run_ex22a(propagator, numsteps=numsteps)[0]))

    plt.clf()
    plt.title("Flow rate vs. density", fontsize=20)
    plt.plot(density_list, flowrate_list, '-x')
    plt.xlabel('density', fontsize=20)
    plt.ylabel('average flow rate', fontsize=20)
    plt.grid()
    plt.savefig("flowrate_vs_density.pdf")
    plt.show()

    max_flowrate = max(flowrate_list)
    max_density = density_list[flowrate_list.index(max_flowrate)]
    print("Max density: ", max_density)


if __name__ == "__main__" :
    exercise22a()


# %% Excercise 2.2 b)

def exercise22b():
    numCars = 25
    vmax = 2
    p = 0.5
    roadLength = 50
    numsteps = 100
    flowrate_list = []

    standard_error_list = []
    #equilibrium_time = []

    n_simulations = 150
    
    for i in range(n_simulations):
        cars = Cars(numCars, roadLength=roadLength)
        sim = Simulation(cars)
        propagator = MyPropagator(vmax, p)
        flowrate, time = sim.run_ex22a(propagator, numsteps=numsteps)
        flowrate_list.append(np.mean(flowrate))

        if i > 0:
            standard_error = np.std(flowrate_list) / np.sqrt(len(flowrate_list))
            standard_error_list.append(standard_error)

    # find equilibrium time, which is the time when the flow rate is constant
        # for j in range(1, len(flowrate)):
        #     if abs(flowrate_list[j] - flowrate_list[j-1]) <= 0.001:
        #         equilibrium_time.append(time[j])
        #         break
    
    for i in range(1, len(standard_error_list)):
        if standard_error_list[i] <= 0.001:
            print("Standard error is less than 0.001 at iteration: ", i)
            break

    plt.clf()
    plt.title("Standard error vs. number of simulations", fontsize=20)
    plt.plot(range(1, n_simulations), standard_error_list, '-x')
    plt.plot(range(1, n_simulations), [0.001 for i in range(1, n_simulations)], '--')
    plt.xlabel('number of simulations', fontsize=20)
    plt.ylabel('standard error', fontsize=20)
    plt.grid()
    plt.savefig("standard_error_vs_num_simulations.pdf")
    plt.show()

    numCars_1 = 25
    numCars_2 = 35
    vmax_1 = 2
    vmax_2 = 4
    p_1 = 0.5
    p_2 = 0.8
    roadLength = 50
    numsteps = 200

    cars_1 = Cars(numCars_1, roadLength=roadLength)
    sim_1 = Simulation(cars_1)
    propagator_1 = MyPropagator(vmax_1, p_1)
    sim_1.run(propagator_1, numsteps=numsteps, title=f"Flowrate for numCars={numCars_1}, vmax={vmax_1}, p={p_1}")

    cars_2 = Cars(numCars_2, roadLength=roadLength)
    sim_2 = Simulation(cars_2)
    propagator_2 = MyPropagator(vmax_1, p_1)
    sim_2.run(propagator_2, numsteps=numsteps, title=f"Flowrate for numCars={numCars_2}, vmax={vmax_1}, p={p_1}")

    cars_3 = Cars(numCars_1, roadLength=roadLength)
    sim_3 = Simulation(cars_3)
    propagator_3 = MyPropagator(vmax_2, p_1)
    sim_3.run(propagator_3, numsteps=numsteps, title=f"Flowrate for numCars={numCars_1}, vmax={vmax_2}, p={p_1}")

    cars_4 = Cars(numCars_1, roadLength=roadLength)
    sim_4 = Simulation(cars_4)
    propagator_4 = MyPropagator(vmax_1, p_2)
    sim_4.run(propagator_4, numsteps=numsteps, title=f"Flowrate for numCars={numCars_1}, vmax={vmax_1}, p={p_2}")

if __name__ == "__main__" :
    exercise22b()

# %% Excercise 2.2 c)

def exercise22c():
    vmax = 2
    p = 0.5
    numsteps = 200
    roadLength = [length for length in range(30, 1000)]
    density = 0.2
    numCars = []
    flowrate_list = []

    for i in range(len(roadLength)):
        numCars.append(int(roadLength[i] * density))
        cars = Cars(numCars[i], roadLength=roadLength[i])
        sim = Simulation(cars)
        propagator = MyPropagator(vmax, p)
        flowrate = sim.run_ex22a(propagator, numsteps=numsteps)[0]
        flowrate_list.append(np.mean(flowrate))

    plt.clf()
    plt.title("Flow rate vs. road length", fontsize=20)
    plt.plot(roadLength, flowrate_list, '-x')
    plt.xlabel('road length', fontsize=20)
    plt.ylabel('average flow rate', fontsize=20)
    plt.grid()
    plt.savefig("flowrate_vs_road_length.pdf")
    plt.show()

if __name__ == "__main__" :
    exercise22c()

# %% Excercise 2.2 d)

def exercise22d():
    vmax = 1
    vmax_1 = 2
    vmax_2 = 5
    p = 0.5
    roadLength = 50
    numsteps = 200
    numCars = 25

    cars = Cars(numCars, roadLength=roadLength)
    sim = Simulation(cars)
    propagator = MyPropagator(vmax, p)
    flowrate = sim.run_ex22a(propagator, numsteps=numsteps)[0]

    cars_1 = Cars(numCars, roadLength=roadLength)
    sim_1 = Simulation(cars_1)
    propagator_1 = MyPropagator(vmax_1, p)
    flowrate_1 = sim_1.run_ex22a(propagator_1, numsteps=numsteps)[0]

    cars_2 = Cars(numCars, roadLength=roadLength)
    sim_2 = Simulation(cars_2)
    propagator_2 = MyPropagator(vmax_2, p)
    flowrate_2 = sim_2.run_ex22a(propagator_2, numsteps=numsteps)[0]

    plt.clf()
    plt.title(f"Flow rate vs. time, velocity = {vmax}", fontsize=20)
    plt.plot(range(numsteps), flowrate, '-x', label=f"vmax={vmax}")
    plt.xlabel('time', fontsize=20)
    plt.ylabel('flow rate', fontsize=20)
    plt.legend()
    plt.grid()
    plt.savefig(f"flowrate_vs_time_velocity{vmax}.pdf")
    plt.show()

    plt.clf()
    plt.title(f"Flow rate vs. time, velocity = {vmax_1}", fontsize=20)
    plt.plot(range(numsteps), flowrate_1, '-x', label=f"vmax={vmax_1}")
    plt.xlabel('time', fontsize=20)
    plt.ylabel('flow rate', fontsize=20)
    plt.legend()
    plt.grid()
    plt.savefig(f"flowrate_vs_time_velocity{vmax_1}.pdf")
    plt.show()

    plt.clf()
    plt.title(f"Flow rate vs. time, velocity = {vmax_2}", fontsize=20)
    plt.plot(range(numsteps), flowrate_2, '-x', label=f"vmax={vmax_2}")
    plt.xlabel('time', fontsize=20)
    plt.ylabel('flow rate', fontsize=20)
    plt.legend()
    plt.grid()
    plt.savefig(f"flowrate_vs_time_velocity{vmax_2}.pdf")
    plt.show()

if __name__ == "__main__" :
    exercise22d()

# %% Excercise 2.2 e)
def exercise22e():
    vmax = 2
    p_1 = 0.5
    p_2 = 0.8
    p_3 = 0.2
    roadLength = 50
    numsteps = 200
    numCars = 25

    cars = Cars(numCars, roadLength=roadLength)
    sim = Simulation(cars)
    propagator = MyPropagator(vmax, p_1)
    flowrate = sim.run_ex22a(propagator, numsteps=numsteps)[0]

    cars_1 = Cars(numCars, roadLength=roadLength)
    sim_1 = Simulation(cars_1)
    propagator_1 = MyPropagator(vmax, p_2)
    flowrate_1 = sim_1.run_ex22a(propagator_1, numsteps=numsteps)[0]

    cars_2 = Cars(numCars, roadLength=roadLength)
    sim_2 = Simulation(cars_2)
    propagator_2 = MyPropagator(vmax, p_3)
    flowrate_2 = sim_2.run_ex22a(propagator_2, numsteps=numsteps)[0]

    plt.clf()
    plt.title(f"Flow rate vs. time, p = {p_1}", fontsize=20)
    plt.plot(range(numsteps), flowrate, '-x', label=f"p={p_1}")
    plt.xlabel('time', fontsize=20)
    plt.ylabel('flow rate', fontsize=20)
    plt.legend()
    plt.grid()
    plt.savefig(f"flowrate_vs_time_p{p_1}.pdf")
    plt.show()
    
    plt.clf()
    plt.title(f"Flow rate vs. time, p = {p_2}", fontsize=20)
    plt.plot(range(numsteps), flowrate_1, '-x', label=f"p={p_2}")
    plt.xlabel('time', fontsize=20)
    plt.ylabel('flow rate', fontsize=20)
    plt.legend()
    plt.grid()
    plt.savefig(f"flowrate_vs_time_p{p_2}.pdf")
    plt.show()

    plt.clf()
    plt.title(f"Flow rate vs. time, p = {p_3}", fontsize=20)
    plt.plot(range(numsteps), flowrate_2, '-x', label=f"p={p_3}")
    plt.xlabel('time', fontsize=20)
    plt.ylabel('flow rate', fontsize=20)
    plt.legend()
    plt.grid()
    plt.savefig(f"flowrate_vs_time_p{p_3}.pdf")
    plt.show()

if __name__ == "__main__" :
    exercise22e()

# %%
