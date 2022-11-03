# %%
# !/bin/python3

# Python simulation of a simple planar pendulum with real time animation
# BH, OF, MP, AJ, TS 2020-10-20, latest version 2022-10-25.

import numpy as np
import matplotlib.pyplot as plt
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
pi = 3.14
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
    def force(self, osc):
        """Virtual method: implemented by the child classes"""
        return NotImplementedError


class Harmonic(BaseSystem):
    def force(self, osc):
        return -osc.m * (osc.c * osc.theta + osc.gamma * osc.dtheta)


class Pendulum(BaseSystem):
    def force(self, osc):
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

        print(f"t = {osc.t:.2f}, theta = {osc.theta:.2f}, dtheta = {osc.dtheta:.2f}")
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
        # accel = simsystem.force(osc) / osc.m
        osc.t += self.dt
        osc.dtheta -= osc.c * osc.theta * self.dt
        osc.theta += osc.dtheta * self.dt


class VerletIntegrator(BaseIntegrator):
    def timestep(self, simsystem, osc, obs):
        accel = simsystem.force(osc) / osc.m
        osc.t += self.dt
        accel_dt = simsystem.force(osc) / osc.m
        osc.theta += osc.dtheta * self.dt + 0.5 * accel * self.dt**2
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
        plt.title(title)
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

        plt.xlabel("time")
        plt.ylabel("observables")
        plt.legend()
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
    integrator = EulerCromerIntegrator()
    tmax = 30
    stepsperframe = 10
    osc = Oscillator(mass, c, time_0, theta_0, dtheta_0, gamma)
    sim = Simulation(osc=osc)
    sim.run(osc, integrator, tmax)
    sim.run_animate(osc, integrator, tmax, stepsperframe=stepsperframe)
    sim.plot_observables("simulation1")


if __name__ == "__main__":
    exercise_11()
# %%
