# %%
import json
import numpy as np
import matplotlib.pyplot as plt
from main import (
    SolarSystem, 
    VelocityVerletIntegrator, 
    Observables, 
    Simulation, 
    plot_trajectories, 
    plot_sum_energies,
    calculate_error
)

start_time = 0
dt = 1
nsteps = 86_400
integrator = VelocityVerletIntegrator(dt)

names = []
colors= []
masses = []
positions = []
velocities = []
accelerations = []

# get data from .json file
# Source data: https://nssdc.gsfc.nasa.gov/planetary/factsheet/ (mass, position, velocity)
for planet in json.load(open("planet_data.json")):
    names.append(planet["name"])
    colors.append(planet["color"])
    masses.append(planet["mass"])
    positions.append(planet["position"])
    velocities.append(planet["velocity"])
    accelerations.append(planet["acceleration"])

masses_solar_system = np.array(masses)
positions_solarsystem = np.array(positions)
velocities_solarsystem = np.array(velocities)
accelerations_solarsystem = np.array(accelerations)

sys_without_asteroid = SolarSystem(start_time, names, colors, masses_solar_system, positions_solarsystem, velocities_solarsystem, accelerations_solarsystem)
obs_without_asteroid = Observables(sys_without_asteroid, nsteps)
sim_without_asteroid = Simulation(sys_without_asteroid, integrator, steps=nsteps, obs=obs_without_asteroid)
sim_without_asteroid.run_simulation()


# %%
names_asteroids = []
colors_asteroids = []
masses_asteroids = []
positions_asteroids = []
velocities_asteroids = [velocity for velocity in range(2, 11, 2)]
accelerations_asteroids = []

displacement = []
error = []

# get asteroid data from .json file
# Source data: https://nssdc.gsfc.nasa.gov/planetary/factsheet/asteroidfact.html
asteroid = json.load(open("asteroid_data.json"))
for asteroid in asteroid:
    names_asteroids.append(asteroid["name"])
    colors_asteroids.append(asteroid["color"])
    masses_asteroids.append(asteroid["mass"])
    positions_asteroids.append(asteroid["position"])
    accelerations_asteroids.append(asteroid["acceleration"])

for i in range(len(names_asteroids)):
    displacement_earth = []
    error_asteroid = []
    for velocity in velocities_asteroids:
        names_solarsystem = names + [names_asteroids[i]]
        colors_solarsystem = colors + [colors_asteroids[i]]
        masses_solar_system = masses + [masses_asteroids[i]]
        positions_solarsystem = positions + [positions_asteroids[i]]
        velocities_solarsystem = velocities + [[velocity, 0, 0]]
        accelerations_solarsystem = accelerations + [accelerations_asteroids[i]]

        masses_solar_system = np.array(masses_solar_system)
        positions_solarsystem = np.array(positions_solarsystem)
        velocities_solarsystem = np.array(velocities_solarsystem)
        accelerations_solarsystem = np.array(accelerations_solarsystem)

        sys_with_asteroid = SolarSystem(start_time, names_solarsystem, colors_solarsystem, masses_solar_system, positions_solarsystem, velocities_solarsystem, accelerations_solarsystem)
        obs_with_asteroid = Observables(sys_with_asteroid, nsteps)
        sim_with_asteroid = Simulation(sys_with_asteroid, integrator, steps=nsteps, obs=obs_with_asteroid)
        sim_with_asteroid.run_simulation()

        # Plot difference between energies
        #plot_sum_energies(obs_with_asteroid)

        # Calculate error
        error_asteroid.append(calculate_error(obs_with_asteroid))

        # Plot difference between trajectories
        obs_with_asteroid.positions[:, :9] = (obs_with_asteroid.positions[:, :9] - obs_without_asteroid.positions) + obs_with_asteroid.positions[0, :9]
        obs_with_asteroid.positions[:, 0:3] = np.zeros((nsteps, 3, 3))
        obs_with_asteroid.positions[:, 4:9] = np.zeros((nsteps, 5, 3))
        #plot_trajectories(obs_with_asteroid)

        # Calculate displacement
        displacement_earth.append(np.sqrt(np.sum(np.square(obs_with_asteroid.positions[-1, 3] - obs_with_asteroid.positions[0, 3]))))

    displacement.append(displacement_earth)
    error.append(error_asteroid)

# %%
# Plot displacement
plt.figure()
for i in range(len(names_asteroids)):
    plt.plot(velocities_asteroids, displacement[i], label=names_asteroids[i], marker="x")
plt.xlabel("Velocity [km/s]")
plt.ylabel("Displacement [km]")
plt.legend()
plt.grid()
plt.savefig("displacement.png")
plt.show()

# %%
# # Plot error
# plt.figure()
# for i in range(len(names_asteroids)):
#     plt.plot(velocities_asteroids, error[i], label=names_asteroids[i])
# plt.xlabel("Velocity [km/s]")
# plt.ylabel("Error")
# plt.legend()
# plt.grid()
# plt.show()