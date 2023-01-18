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
class SolarSystem:

    def __init__(
        self, 
        start_time: float, 
        names: list, 
        colors: list, 
        masses: ndarray,
        positions: ndarray, 
        velocities: ndarray, 
        accelerations: ndarray
    ) -> None:
        self.time = start_time
        self.names = names
        self.colors = colors
        self.masses = masses
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations

    def update_accelerations(self) -> None:
        """Calculate the forces acting on each body"""
        dists = self.positions[:, None, :] - self.positions[None, :, :]  # (N, N, 3)
        r = np.sqrt(np.sum(dists ** 2, axis=-1))  # (N, N)
        a = G * self.masses[:, None] / (r ** 3)  # (N, N)
        a = np.where(r == 0, 0, a)  # (N, N)
        self.accelerations = np.sum(a[:, :, None] * dists, axis=0)  # (N, 3)


class Observables:

    def __init__(self, solar_system: SolarSystem, nsteps: int) -> None:
        self.names = solar_system.names  # names
        self.colors = solar_system.colors  # colors
        nbodies = len(solar_system.names)  # number of bodies
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

        # Append position, velocity, kinetic, potential and total energy observables to their arrays
        obs.positions[current_step] = solar_system.positions
        obs.velocities[current_step] = solar_system.velocities

        k = 0.5 * solar_system.masses * np.sum(np.square(solar_system.velocities), axis=1)  # (N)

        dists = solar_system.positions[:, None, :] - solar_system.positions[None, :, :]  # (N, N, 3)
        r = np.sqrt(np.sum(np.square(dists), axis=-1))  # (N, N)
        p = -G * solar_system.masses[:, None] * solar_system.masses[None, :] / r  # (N, N)
        p = np.where(r == 0, 0, p)  # (N, N)

        obs.kinetic_energy[current_step] = k
        obs.potential_energy[current_step] = np.sum(p, axis=1)
        obs.total_energy[current_step] = obs.kinetic_energy[current_step] + obs.potential_energy[current_step]


class EulerCromerIntegrator(BaseIntegrator):

    def timestep(self, sys: SolarSystem) -> None:
        sys.update_accelerations()
        sys.velocities = sys.velocities + sys.accelerations * self.dt
        sys.positions = sys.positions + sys.velocities * self.dt


class VelocityVerletIntegrator(BaseIntegrator):

    def timestep(self, sys: SolarSystem) -> None:
        sys.update_accelerations()
        accelerations_old = sys.accelerations
        sys.positions = sys.positions + sys.velocities * self.dt + 0.5 * sys.accelerations * self.dt ** 2
        sys.update_accelerations()
        sys.velocities = sys.velocities + 0.5 * (accelerations_old + sys.accelerations) * self.dt


class LeapFrogIntegrator(BaseIntegrator):

    def timestep(self, sys: SolarSystem) -> None:
        sys.update_accelerations()
        sys.velocities = sys.velocities + sys.accelerations * self.dt / 2
        sys.positions = sys.positions + sys.velocities * self.dt
        sys.update_accelerations()
        sys.velocities = sys.velocities + sys.accelerations * self.dt / 2

  
# Simulation
class Simulation:

    def __init__(self, sys: SolarSystem, integrator: BaseIntegrator, steps: int, obs: Observables) -> None:
        self.solar_system = sys
        self.integrator = integrator
        self.steps = steps
        self.obs = obs

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

# Source data: https://nssdc.gsfc.nasa.gov/planetary/factsheet/ (mass, position, velocity)
dt = 100_000
nsteps = 10_000

names = []
colors = []
masses = []
positions = []
velocities = []
accelerations = []

# get data from .json file
for planet in json.load(open("planet_data.json")):
    names.append(planet["name"])
    colors.append(planet["color"])
    masses.append(planet["mass"])
    positions.append(planet["position"])
    velocities.append(planet["velocity"])
    accelerations.append(planet["acceleration"])

masses = np.array(masses)
positions = np.array(positions)
velocities = np.array(velocities)
accelerations = np.array(accelerations)

sys = SolarSystem(start_time, names, colors, masses, positions, velocities, accelerations)
integrator = LeapFrogIntegrator(dt)
obs = Observables(sys, nsteps)
sim = Simulation(sys, integrator, steps=nsteps, obs=obs)
sim.run_simulation()

# Plot trajectories
plot_trajectories(obs)

# Plot energies
plot_sum_energies(obs)