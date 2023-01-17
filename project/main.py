# %% Project SimMod
#
# Simulate our solar system, add an asteroid and make it pass very close
# (such that it’s direction changes significantly) by a planet, you can also
# vary the mass of the asteroid from realistic to planet like. Analyze by
# how much the trajectories of the planets and asteroid change.

# Imports
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm

from numpy import ndarray
from typing import List
from abc import abstractmethod, ABC

# %% Global constants
G = 6.6740831 * (10 ** (-20))  # Gravitational constant [km^3 kg^-1 s^-2]

# %% Main functions
class Body:

    def __init__(
        self, 
        name: str,
        color: str,
        mass: float,
        position: ndarray = np.array([0, 0, 0]),
        velocity: ndarray = np.array([0, 0, 0]),
        acceleration: ndarray = np.array([0, 0, 0]),
        time: float = 0,
    ) -> None:
        self.name = name
        self.color = color
        self.time = time
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration


class SolarSystem:

    def __init__(self, bodies: List[Body], start_time: float = 0) -> None:
        self.bodies = bodies
        self.time = start_time

    def forces(self) -> None:
        for body in self.bodies:
            acceleration = np.zeros_like(body.acceleration)
            for other in self.bodies:
                if body.name == other.name:
                    continue
                r = other.position - body.position
                force = (
                    G 
                    * body.mass
                    * other.mass
                    / (np.sum(np.square(r)) ** (3 / 2))
                    * r
                )
                acceleration = acceleration + force / body.mass
            body.acceleration = acceleration


class Observables:

    def __init__(self, solar_system: SolarSystem, nsteps: int) -> None:
        self.names = [body.name for body in solar_system.bodies]  # names
        self.colors = [body.color for body in solar_system.bodies]  # colors
        nbodies = len(solar_system.bodies)  # number of bodies
        self.time = np.zeros(nsteps)  # array to store time
        self.positions = np.zeros((nsteps, nbodies, 3))  # array to store positions
        self.velocities = np.zeros((nsteps, nbodies, 3))  # array to store velocities
        self.kinetic_energy = np.zeros((nsteps, nbodies))  # array to store kinetic energies
        self.potential_energy = np.zeros((nsteps, nbodies))  # array to store potential energies
        self.total_energy = np.zeros((nsteps, nbodies))  # array to store total energies


# %% Integrators
class BaseIntegrator(ABC):

    def __init__(self, dt: float=0.01):
        self.dt = dt  # time step

    @abstractmethod
    def timestep(self, solar_system: SolarSystem, obs: Observables) -> None:
        """Virtual method: implemented by the child classes"""
        return NotImplementedError

    def integrate(self, solar_system: SolarSystem, obs: Observables, current_step: int) -> None:
        """Perform a single integration step"""
        # calculate energies before integrating

        self.timestep(solar_system)

        # Append time observables to their lists
        solar_system.time += self.dt
        obs.time[current_step] = solar_system.time



        for body_index, body in enumerate(solar_system.bodies):
            kinetic = 0.5 * body.mass * np.sum(np.square(body.velocity))
            potential = 0
            for other in solar_system.bodies:
                if body == other:
                    continue
                r = other.position - body.position
                potential = potential - G * body.mass * other.mass / np.sqrt(np.sum(np.square(r)))

            obs.positions[current_step, body_index] = body.position
            obs.velocities[current_step, body_index] = body.velocity
            obs.kinetic_energy[current_step, body_index] = kinetic
            obs.potential_energy[current_step, body_index] = potential
            obs.total_energy[current_step, body_index] = kinetic + potential


class EulerCromerIntegrator(BaseIntegrator):

    def timestep(self, solar_system) -> None:
        solar_system.forces()
        for body in solar_system.bodies:
            body.velocity = body.velocity + body.acceleration * self.dt
            body.position = body.position + body.velocity * self.dt


class VelocityVerletIntegrator(BaseIntegrator):

    def timestep(self, solar_system) -> None:
        solar_system.forces()
        for body in solar_system.bodies:
            acceleration = body.acceleration
            body.position = body.position + body.velocity * self.dt + 0.5 * body.acceleration * self.dt ** 2
            body.velocity = body.velocity + 0.5 * (body.acceleration + acceleration) * self.dt


class LeapFrogIntegrator(BaseIntegrator):

    def timestep(self, solar_system) -> None:
        solar_system.forces()
        for body in solar_system.bodies:
            body.velocity = body.velocity + body.acceleration * self.dt / 2
            body.position = body.position + body.velocity * self.dt
            solar_system.forces()
            body.velocity = body.velocity + body.acceleration * self.dt / 2
   
# %% Simulation
class Simulation:

    def __init__(self, solar_system: SolarSystem, integrator: BaseIntegrator, steps: int, obs: Observables) -> None:
        self.solar_system = solar_system
        self.integrator = integrator
        self.steps = steps
        self.obs = obs

    # Plot the animation of the solar system using matplotlib and the observables
    def run_and_plot_simulation(self) -> None:
        fig = plt.figure()
        size = 1e10
        ax = plt.axes(xlim=(-size, size), ylim=(-size, size))
        ax.set_aspect("equal")
        ax.grid()
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_title("The solar system")

        lines = []
        for body in self.solar_system.bodies:
            line, = ax.plot([], [], "o", color=body.color, label=body.name)
            lines.append(line)

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(i):
            self.integrator.integrate(self.solar_system, self.obs, i)
            for line, body in zip(lines, self.solar_system.bodies):
                line.set_data(body.position[0], body.position[1])
            return lines

        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=self.steps, interval=1, blit=True, 
        )
        plt.legend()
        plt.show()

    def run_simulation(self) -> None:
        for i in tqdm.trange(self.steps):
            self.integrator.integrate(self.solar_system, self.obs, i)

# %% Plot trajectories for all bodies

def plot_trajectories(obs: Observables) -> None:
    plt.figure()
    plt.title("Trajectories")
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    for i, name in enumerate(obs.names):
        x = obs.positions[:, i, 0]
        y = obs.positions[:, i, 1]
        plt.plot(x, y, label=name)
    plt.grid()
    plt.legend()
    plt.show()

# %% Plot energies for all bodies

def plot_sum_energies(obs: Observables) -> None:
    plt.figure()
    plt.title("Kinetic, Potential and Total Energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.plot(obs.time, np.sum(obs.kinetic_energy, axis=1), label="Kinetic energy", color="red")
    plt.plot(obs.time, np.sum(obs.potential_energy, axis=1), label="Potential energy", color="blue")
    plt.plot(obs.time, np.sum(obs.total_energy, axis=1), label="Total energy", color="green")
    plt.grid()
    plt.legend()
    plt.show()

# %% Main
# Run simulation
start_time = 0
nsteps = 1000

# Source data: https://nssdc.gsfc.nasa.gov/planetary/factsheet/ (mass, position, velocity)
bodies = [
    Body(
        name=data["name"],
        color=data["color"],
        mass=data["mass"],
        position=np.array(data["position"], dtype=np.float64),
        velocity=np.array(data["velocity"], dtype=np.float64),
    ) for data in json.load(open('/Users/cecilie/Desktop/Skrivbord – Cecilies MacBook Air/Universitet/tredje/simmod/project/planet_data.json'))
]

sys = SolarSystem(bodies, start_time)
integrator = LeapFrogIntegrator(dt=100_000)
obs = Observables(sys, nsteps)
sim = Simulation(sys, integrator, steps=nsteps, obs=obs)
sim.run_simulation()

# Plot trajectories
plot_trajectories(obs)

# Plot energies
plot_sum_energies(obs)