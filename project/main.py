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

    def __init__(self):
        self.time = []  # list to store time
        self.positions = []  # list to store positions
        self.velocities = []  # list to store velocities

        self.kinetic_energy = []  # list to store kinetic energy
        self.potential_energy = []  # list to store potential energy
        self.total_energy = []  # list to store total energy

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
        self.timestep(solar_system)

        # Append observables to their lists
        solar_system.time += self.dt
        obs.time.append(solar_system.time)

        positions = []
        velocities = []
        kinetic_energy = []
        potential_energy = []
        total_energy = []
        for body in solar_system.bodies:
            positions.append(body.position)
            velocities.append(body.velocity)
            if body.name != 'Sun':
                kinetic_energy.append(0.5 * body.mass * np.sum(np.square(body.velocity)))
                potential_energy.append(-G * body.mass * solar_system.bodies[0].mass / np.sqrt(np.sum(np.square(body.position - solar_system.bodies[0].position))))
                total_energy.append(kinetic_energy[-1] + potential_energy[-1])
            
        obs.positions.append(positions)
        obs.velocities.append(velocities)
        obs.kinetic_energy.append(kinetic_energy)
        obs.potential_energy.append(potential_energy)
        obs.total_energy.append(total_energy)


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
            body.position = body.position + body.velocity * self.dt + 0.5 * body.acceleration * self.dt ** 2
            body.velocity = body.velocity + 0.5 * (body.acceleration + solar_system.acceleration) * self.dt


class RungeKuttaIntegrator(BaseIntegrator):

    def timestep(self, solar_system) -> None:
        solar_system.forces()
        for body in solar_system.bodies:
            k1 = body.acceleration
            k2 = (body.acceleration + k1 * self.dt / 2) / (self.dt / 2)
            k3 = (body.acceleration + k2 * self.dt / 2) / (self.dt / 2)
            k4 = (body.acceleration + k3 * self.dt) / self.dt
            body.velocity = body.velocity + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
            body.position = body.position + body.velocity * self.dt


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
        size = 64983964400
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
            fig, animate, init_func=init, frames=self.steps, interval=20, blit=True, 
        )
        plt.legend()
        plt.show()


# Run simulation
start_time = 0

# Source data: https://ssd.jpl.nasa.gov/horizons/app.html#/ (mass, position, velocity) (82459928.500000000 = A.D. 2022-Dec-15 00:00:00.0000 TDB)
bodies = [
    Body(
        name=data["name"],
        color=data["color"],
        mass=data["mass"] * 10 ** 24,
        position=np.array(data["position"], dtype=np.float64) * 10 ** 7,
        velocity=np.array(data["velocity"], dtype=np.float64),
    ) for data in json.load(open('/Users/cecilie/Desktop/Skrivbord – Cecilies MacBook Air/Universitet/tredje/simmod/project/planet_data.json'))
]

sys = SolarSystem(bodies, start_time)
integrator = EulerCromerIntegrator(dt=100_000_000)
obs = Observables()
sim = Simulation(sys, integrator, number_of_steps_per_frame=1, steps=100_000_000_000_000, obs=obs)
sim.plot_simulation()
