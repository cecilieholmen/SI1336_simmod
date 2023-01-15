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

    def __init__(self, solar_system: SolarSystem) -> None:
        self.names = [body.name for body in solar_system.bodies]  # names
        self.colors = [body.color for body in solar_system.bodies]  # colors
        self.time = []  # list to store time
        self.positions = []  # list to store positions
        self.velocities = []  # list to store velocities
        self.kinetic_energy = []  # list to store kinetic energies
        self.potential_energy = []  # list to store potential energies
        self.total_energy = []  # list to store total energies


# %% Integrators
class BaseIntegrator(ABC):

    def __init__(self, dt: float=0.01):
        self.dt = dt  # time step

    @abstractmethod
    def timestep(self, solar_system: SolarSystem, obs: Observables) -> None:
        """Virtual method: implemented by the child classes"""
        return NotImplementedError

    def integrate(self, solar_system, obs) -> None:
        """Perform a single integration step"""
        # calculate energies before integrating

        self.timestep(solar_system)

        # Append time observables to their lists
        solar_system.time += self.dt
        obs.time.append(solar_system.time)

        positions = []
        velocities = []
        potential_energies = []
        kinetic_energies = []
        total_energies = []
        for body in solar_system.bodies:
            kinetic = 0.5 * body.mass * np.sum(np.square(body.velocity))
            potential = 0
            for other in solar_system.bodies:
                if body == other:
                    continue
                r = other.position - body.position
                potential = potential - G * body.mass * other.mass / np.sqrt(np.sum(np.square(r)))

            positions.append(body.position)
            velocities.append(body.velocity)
            kinetic_energies.append(kinetic)
            potential_energies.append(potential)
            total_energies.append(kinetic + potential)

        obs.positions.append(positions)
        obs.velocities.append(velocities)
        obs.kinetic_energy.append(kinetic_energies)
        obs.potential_energy.append(potential_energies)
        obs.total_energy.append(total_energies)


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

    def __init__(self, solar_system: SolarSystem, integrator: BaseIntegrator, number_of_steps_per_frame: int, steps: int, obs: Observables) -> None:
        self.solar_system = solar_system
        self.integrator = integrator
        self.number_of_steps_per_frame = number_of_steps_per_frame
        self.steps = steps
        self.obs = obs

    # Plot the animation of the solar system using matplotlib and the observables
    def plot_simulation(self) -> None:
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
            self.integrator.integrate(self.solar_system, self.obs)
            for line, body in zip(lines, self.solar_system.bodies):
                line.set_data(body.position[0], body.position[1])
            return lines

        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=self.steps, interval=1, blit=True, 
        )
        plt.legend()
        plt.show()

# %% Plot energies for all bodies

def plot_kinetic_energy(obs: Observables) -> None:
    plt.figure()
    plt.title("Kinetic energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Kinetic energy [J]")
    for i, name in enumerate(obs.names):
        plt.plot(obs.time, [l[i] for l in obs.kinetic_energy], label=name, color=obs.colors[i])
    plt.grid()
    plt.legend()
    plt.show()

def plot_potential_energy(obs: Observables) -> None:
    plt.figure()
    plt.title("Potential energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Potential energy [J]")
    for i, name in enumerate(obs.names):
        plt.plot(obs.time, [l[i] for l in obs.potential_energy], label=name, color=obs.colors[i])
    plt.grid()
    plt.legend()
    plt.show()

def plot_total_energy(obs: Observables) -> None:
    plt.figure()
    plt.title("Total energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Total energy [J]")
    for i, name in enumerate(obs.names):
        plt.plot(obs.time, [l[i] for l in obs.total_energy], label=name, color=obs.colors[i])
    plt.grid()
    plt.legend()
    plt.show()

# %% Main
# Run simulation
start_time = 0

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
integrator = EulerCromerIntegrator(dt=100_000)
obs = Observables(sys)
sim = Simulation(sys, integrator, number_of_steps_per_frame=1, steps=100_000_000, obs=obs)
sim.plot_simulation()

# Plot energies
plot_kinetic_energy(obs)
plot_potential_energy(obs)
plot_total_energy(obs)