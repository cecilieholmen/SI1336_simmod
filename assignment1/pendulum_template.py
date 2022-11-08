# %%
# !/bin/python3

# Python simulation of a simple planar pendulum with real time animation
# BH, OF, MP, AJ, TS 2020-10-20, latest version 2022-10-25.

import numpy as np
import matplotlib.pyplot as plt
import math as math
from matplotlib import animation

from abc import abstractmethod

"""
    This script defines all the classes needed to simulate (and animate) a single pendulum.
    Hierarchy (somehow in order of encapsulation):
    - Oscillator: a struct that stores the parameters of an oscillator (harmonic or pendulum)
    - Observable: a struct that stores the oscillator's coordinates and energy values over time
    - BaseSystem: harmonic oscillators and pendolums are distinguished only by the expression of
                    the return force. This base class defines a virtual force method, which is
                    specified by its child classes
                    -> Harmonic: specifies the return force as -k*t (i.e. spring)
                    -> Pendulum: specifies the return force as -k*sin(t)
    - BaseIntegrator: parent class for all time-marching schemes; function integrate performs
                    a numerical integration steps and updates the quantity of the system provided
                    as input; function timestep wraps the numerical scheme itself and it's not
                    directly implemented by BaseIntegrator, you need to implement it in his child
                    classes (names are self-explanatory)
                    -> EulerCromerIntegrator: ...
                    -> VerletIntegrator: ...
                    -> RK4Integrator: ...
    - Simulation: this last class encapsulates the whole simulation procedure; functions are 
                    self-explanatory; you can decide whether to just run the simulation or to
                    run while also producing an animation: the latter option is slower
"""

G = 9.8  # gravitational acceleration
pi = np.pi
cos = np.cos
sin = np.sin

# %% Classes
class Oscillator:

    """Class for a general, simple oscillator"""

    def __init__(self, m=1, c=4, t0=0, theta0=0, dtheta0=0, gamma=0):
        self.m = m  # mass of the pendulum bob
        self.c = c  # c = g/L
        self.L = G / c  # string length
        self.t = t0  # the time
        self.theta = theta0  # the position/angle
        self.dtheta = dtheta0  # the velocity
        self.gamma = gamma  # damping


class Observables:

    """Class for storing observables for an oscillator"""

    def __init__(self):
        self.time = []  # list to store time
        self.pos = []  # list to store positions
        self.vel = []  # list to store velocities
        self.energy = []  # list to store energy


class BaseSystem:
    @abstractmethod
    def force(osc):
        """Virtual method: implemented by the child classes"""
        return NotImplementedError


class Harmonic(BaseSystem):
    @staticmethod
    def force(osc):
        return -osc.m * (osc.c * osc.theta + osc.gamma * osc.dtheta)


class Pendulum(BaseSystem):
    @staticmethod
    def force(osc):
        return -osc.m * (osc.c * np.sin(osc.theta) + osc.gamma * osc.dtheta)


class BaseIntegrator:
    def __init__(self, _dt=0.01):
        self.dt = _dt  # time step

    @abstractmethod
    def timestep(self, simsystem, osc, obs):
        """Virtual method: implemented by the child classes"""
        return NotImplementedError

    def integrate(self, simsystem, osc, obs):
        """Perform a single integration step"""
        self.timestep(simsystem, osc, obs)

        # Append observables to their lists
        obs.time.append(osc.t)
        obs.pos.append(osc.theta)
        obs.vel.append(osc.dtheta)
        # Function 'isinstance' is used to check if the instance of the system object is 'Harmonic' or 'Pendulum'
        if isinstance(simsystem, Harmonic):
            # Harmonic oscillator energy
            obs.energy.append(
                0.5 * osc.m * osc.L**2 * osc.dtheta**2
                + 0.5 * osc.m * G * osc.L * osc.theta**2
            )
        else:
            # Pendulum energy
            obs.energy.append(
                0.5 * osc.m * osc.L**2 * osc.dtheta**2
                + osc.m * G * osc.L * (1 - cos(osc.theta))
            )


class EulerCromerIntegrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        osc.t += self.dt
        osc.dtheta -= osc.c * osc.theta * self.dt
        osc.theta += osc.dtheta * self.dt


class VerletIntegrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        accel = simsystem.force(osc) / osc.m
        osc.t += self.dt
        osc.theta += osc.dtheta * self.dt + 0.5 * accel * self.dt**2
        accel_dt = simsystem.force(osc) / osc.m
        osc.dtheta += 0.5 * (accel_dt + accel) * self.dt


class RK4Integrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        theta = osc.theta
        dtheta = osc.dtheta

        a_1 = simsystem.force(osc) / osc.m * self.dt
        b_1 = osc.dtheta * self.dt

        osc.t += self.dt / 2
        osc.theta += b_1 / 2
        osc.dtheta += a_1 / 2
        a_2 = simsystem.force(osc) / osc.m * self.dt
        b_2 = osc.dtheta * self.dt

        osc.theta = theta + b_2 / 2
        osc.dtheta = dtheta + a_2 / 2
        a_3 = simsystem.force(osc) / osc.m * self.dt
        b_3 = osc.dtheta * self.dt

        osc.t += self.dt / 2 
        osc.theta = theta + b_3
        osc.dtheta = dtheta + a_3
        a_4 = simsystem.force(osc) / osc.m * self.dt
        b_4 = osc.dtheta * self.dt

        osc.dtheta = dtheta + 1 / 6 * (a_1 + 2 * a_2 + 2 * a_3 + a_4)
        osc.theta = theta + 1 / 6 * (b_1 + 2 * b_2 + 2 * b_3 + b_4)


# Animation function which integrates a few steps and return a line for the pendulum
def animate(
    framenr, simsystem, oscillator, obs, integrator, pendulum_line, stepsperframe
):

    for it in range(stepsperframe):
        integrator.integrate(simsystem, oscillator, obs)

    x = np.array([0, np.sin(oscillator.theta)])
    y = np.array([0, -np.cos(oscillator.theta)])
    pendulum_line.set_data(x, y)
    return pendulum_line


# Simulation
class Simulation:
    def reset(self, osc=Oscillator()):
        self.oscillator = osc
        self.obs = Observables()

    def __init__(self, osc=Oscillator()):
        self.reset(osc)

    # Run without displaying any animation (fast)

    def run(
        self,
        simsystem,
        integrator,
        tmax=30.0,  # final time
    ):

        n = int(tmax / integrator.dt)
        for it in range(n):
            integrator.integrate(simsystem, self.oscillator, self.obs)

    # Run while displaying the animation of a pendulum swinging back and forth (slow-ish)
    # If too slow, try to increase stepsperframe
    def run_animate(
        self,
        simsystem,
        integrator,
        tmax=30.0,  # final time
        stepsperframe=1,  # how many integration steps between visualising frames
        title='Animation'
    ):

        numframes = int(tmax / (stepsperframe * integrator.dt)) - 2

        # WARNING! If you experience problems visualizing the animation try to comment/uncomment this line
        plt.clf()

        # If you experience problems visualizing the animation try to comment/uncomment this line
        # fig = plt.figure()

        ax = plt.subplot(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        plt.axhline(y=0)  # draw a default hline at y=1 that spans the xrange
        plt.axvline(x=0)  # draw a default vline at x=1 that spans the yrange
        (pendulum_line,) = ax.plot([], [], lw=5)
        plt.title(title)
        # Call the animator, blit=True means only re-draw parts that have changed
        anim = animation.FuncAnimation(
            plt.gcf(),
            animate,  # init_func=init,
            fargs=[
                simsystem,
                self.oscillator,
                self.obs,
                integrator,
                pendulum_line,
                stepsperframe,
            ],
            frames=numframes,
            interval=25,
            blit=False,
            repeat=False,
        )

        # If you experience problems visualizing the animation try to comment/uncomment this line
        plt.show()

        # If you experience problems visualizing the animation try to comment/uncomment this line
        # plt.waitforbuttonpress(10)

    # Plot coordinates and energies (to be called after running)
    def plot_observables(self, title="simulation", ref_E=None):

        plt.clf()
        plt.title(title, fontdict = {'fontsize' : 30})
        plt.plot(self.obs.time, self.obs.pos, "b-", label="Position")
        plt.plot(self.obs.time, self.obs.vel, "r-", label="Velocity")
        plt.plot(self.obs.time, self.obs.energy, "g-", label="Energy")
        if ref_E != None:
            plt.plot(
                [self.obs.time[0], self.obs.time[-1]],
                [ref_E, ref_E],
                "k--",
                label="Ref.",
            )

        plt.xlabel("time", fontsize = 20)
        plt.ylabel("observables", fontsize = 20)
        plt.legend(fontsize = 20)
        plt.savefig(title + ".png")
        plt.show()


# %% Exercise 1.1
def exercise_11():
    # Compare the different methods, Euler-Cromer, velocity Verlet and Runge-Kutta, with each other and with the exact solution of the harmonic oscillator
    mass = 1
    c = 2**2
    time_0 = 0
    theta_0 = 0.1 * pi
    dtheta_0 = 0
    gamma = 0
    dt = 0.01
    integrator1 = EulerCromerIntegrator(dt)
    integrator2 = VerletIntegrator(dt)
    integrator3 = RK4Integrator(dt)
    system1 = Harmonic()
    system2 = Pendulum()
    tmax = 30
    stepsperframe = 10

    osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
    sim = Simulation(osc=osc)
    sim.run(system1, integrator1, tmax)
    #sim.run_animate(osc, integrator1, tmax, stepsperframe=stepsperframe)
    sim.plot_observables("Euler-Cromer, Harmonic osc.")
    osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
    sim = Simulation(osc=osc)
    sim.run(system2, integrator1, tmax)
    #sim.run_animate(osc, integrator1, tmax, stepsperframe=stepsperframe)
    sim.plot_observables("Euler-Cromer, Pendulum")

    osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
    sim = Simulation(osc=osc)
    sim.run(system1, integrator2, tmax)
    #sim.run_animate(system, integrator2, tmax, stepsperframe=stepsperframe)
    sim.plot_observables("Verlet with Harmonic oscillator")
    osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
    sim = Simulation(osc=osc)
    sim.run(system2, integrator2, tmax)
    #sim.run_animate(osc, integrator1, tmax, stepsperframe=stepsperframe)
    sim.plot_observables("Verlet with Pendulum")

    osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
    sim = Simulation(osc=osc)
    sim.run(system1, integrator3, tmax)
    #sim.run_animate(system, integrator3, tmax, stepsperframe=stepsperframe)
    sim.plot_observables("RK4 with Harmonic oscillator")
    osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
    sim = Simulation(osc=osc)
    sim.run(system2, integrator3, tmax)
    #sim.run_animate(osc, integrator1, tmax, stepsperframe=stepsperframe)
    sim.plot_observables("RK4 with Pendulum")

    dt1 = 0.01
    dt2 = 0.1
    dt3 = 0.2
    integrator_dt1 = VerletIntegrator(dt1)
    osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
    sim = Simulation(osc=osc)
    sim.run(system2, integrator_dt1, tmax)
    sim.plot_observables("Verlet with Pendulum, time step 0.01")
    integrator_dt2 = VerletIntegrator(dt2)
    osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
    sim = Simulation(osc=osc)
    sim.run(system2, integrator_dt2, tmax)
    sim.plot_observables("Verlet with Pendulum, time step 0.1")
    integrator_dt3 = VerletIntegrator(dt3)
    osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
    sim = Simulation(osc=osc)
    sim.run(system2, integrator_dt3, tmax)
    sim.plot_observables("Verlet with Pendulum, time step 0.2")


if __name__ == "__main__":
    exercise_11()

# %% Exercise 1.2
def exercise_12():
    # Determine the period time T as a function of the initial position theta_0 and determine which system (harmonic osc./pendulum) has a larger period
    mass = 1
    c = 2**2
    time_0 = 0
    dtheta_0 = 0
    gamma = 0
    dt = 0.01
    integrator1 = EulerCromerIntegrator(dt)
    system1 = Harmonic()
    system2 = Pendulum()
    tmax = 30

    T_harmonic = []
    T_pendulum = []
    T_pertutbulation_series = [] # the perturbation series to compare with the pendulum
    theta_0_series = []

    for i in range(1, 10):
        theta_0 = 0.1 * i * pi
        theta_0_series.append(theta_0)
        T_pertutbulation_series.append(2 * pi * math.sqrt(c * (1 + theta_0**2 / 16 + 11 * theta_0**4 / 3072 + 173 * theta_0**6 / 737280))) 


        osc1 = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
        sim1 = Simulation(osc=osc1)
        sim1.run(system1, integrator1, tmax)

        local_max_harmonic = []
        for j in range(1, len(sim1.obs.pos)-1):
            if sim1.obs.pos[j] > sim1.obs.pos[j-1] and sim1.obs.pos[j] > sim1.obs.pos[j+1]:
                local_max_harmonic.append(sim1.obs.time[j])
        T_harmonic.append(local_max_harmonic[1] - local_max_harmonic[0])


        osc2 = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
        sim2 = Simulation(osc=osc2)
        sim2.run(system2, integrator1, tmax)

        local_max_pendulum = []
        for k in range(1, len(sim2.obs.pos)-1):
            if sim2.obs.pos[k] > sim2.obs.pos[k-1] and sim2.obs.pos[k] > sim2.obs.pos[k+1]:
                local_max_pendulum.append(sim2.obs.time[k])
        T_pendulum.append(local_max_pendulum[1] - local_max_pendulum[0])

    plt.title('Period time as a function of theta(0)', fontdict = {'fontsize' : 30})
    plt.plot(theta_0_series, T_pendulum, label="Pendulum")
    plt.plot(theta_0_series, T_harmonic, label="Harmonic oscillator")
    plt.plot(theta_0_series, T_pertutbulation_series, label="Perturbation series")
    plt.xlabel("theta_0", fontsize=20)
    plt.ylabel("Period time T", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("PeriodTime12.png")
    plt.show()


if __name__ == "__main__":
    exercise_12()

# %% Exercise 1.3

def exercise_13():
    # Study the damped harmonic oscillator and estimate the relaxation time, tau. Study the dependence of tau on gamma. Find the smallest gamma such that the oscillator does not pass theta = 0.
    mass = 1
    omega0 = 2 
    c = omega0 ** 2
    gamma1 = 0.5
    time_0 = 0
    theta_0 = 1 
    dtheta_0 = 0
    dt = 0.01
    tmax = 30
    stepsperframe = 10

    integrator = VerletIntegrator(dt)
    system = Harmonic()      

    osc1 = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma1)
    sim1 = Simulation(osc=osc1)
    sim1.run(system, integrator, tmax)
    sim1.plot_observables("Damped harmonic oscillator, gamma = 0.5")

    tau_list = []
    gamma_list = [gamma for gamma in np.linspace(0.5, 3, num=15)]
    for gamma in np.linspace(0.5, 3, num=15):
        osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
        sim = Simulation(osc=osc)
        sim.run(system, integrator, tmax)

        amplitude_time = []
        for j in range(1, len(sim.obs.pos)-1):
            if sim.obs.pos[j] > sim.obs.pos[j-1] and sim.obs.pos[j] > sim.obs.pos[j+1]:
                amplitude_time.append(sim.obs.time[j])

        reduced_amplitude = []       
        for k in range(1, len(sim.obs.pos)-1):
            if sim.obs.pos[k] < 1.5 * 0.37 * amplitude_time[0] and sim.obs.pos[k] > 0.5 * 0.37 * amplitude_time[0]:
                reduced_amplitude.append(sim.obs.time[k])
            
        tau_list.append(reduced_amplitude[-1])
    
    plt.title('Dependence of tau on gamma', fontdict = {'fontsize' : 30})
    plt.plot(gamma_list, tau_list, label="Dependence of tau on gamma")
    plt.xlabel("Gamma", fontsize=20)
    plt.ylabel("Tau", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("GammaTau13.png")
    plt.show()

    for gamma_critical in np.linspace(3, 11, num=30):
        osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma_critical)
        sim = Simulation(osc=osc)
        sim.run(system, integrator, tmax)
        if min(sim.obs.pos) >= 0:
            print(gamma_critical) # we get gamma_critical = 4.103448275862069
            break


if __name__ == "__main__":
    exercise_13()

# %% Exercise 1.4

def exercise_14():
    # Study the damped pendulum. Determine the phase space portrait. 
    mass = 1
    c = 2 ** 2
    gamma = 1
    time_0 = 0
    theta_0 = pi / 2
    dtheta_0 = 0
    dt = 0.01
    tmax = 30
    stepsperframe = 10

    integrator = VerletIntegrator(dt)
    system = Pendulum()  

    osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
    sim = Simulation(osc=osc)
    sim.run(system, integrator, tmax)

    # plot dtheta vs theta 
    plt.title('Phase space portrait', fontdict = {'fontsize' : 30})
    plt.plot(sim.obs.pos, sim.obs.vel, label="Phase space portrait")
    plt.xlabel("Theta", fontsize=20)
    plt.ylabel("dTheta", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("PhaseSpacePortrait.png")
    plt.show()

if __name__ == "__main__":
    exercise_14()