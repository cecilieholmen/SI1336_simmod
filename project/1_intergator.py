# %%
import json
import numpy as np
from main import (
    SolarSystem, 
    LeapFrogIntegrator, 
    VelocityVerletIntegrator,
    EulerCromerIntegrator,
    Observables, 
    Simulation, 
    plot_trajectories, 
    plot_sum_energies,
    calculate_error
)
import matplotlib.pyplot as plt

names = []
colors = []
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

masses = np.array(masses)
positions = np.array(positions)
velocities = np.array(velocities)
accelerations = np.array(accelerations)

# %% plot the error for different integrators and different number of steps
start_time = 0
total_time = 1e9
Integrators = [LeapFrogIntegrator, VelocityVerletIntegrator, EulerCromerIntegrator]
nsteps = [1_000, 5_000, 10_000, 50_000, 100_000]

error = []

for Integrator in Integrators:
    error_integrator = []
    for steps in nsteps:
        dt = total_time / steps
        sys = SolarSystem(start_time, names, colors, masses, positions, velocities, accelerations)
        integrator = Integrator(dt)
        obs = Observables(sys, steps)
        sim = Simulation(sys, integrator, steps=steps, obs=obs)
        sim.run_simulation()
        error_integrator.append(calculate_error(obs))

    error.append(error_integrator)

# %%

dt = []
for steps in nsteps:
    dt.append(total_time / steps)

plt.figure()
plt.plot(dt, error[0], label="LeapFrogIntegrator")
plt.plot(dt, error[1], label="VelocityVerletIntegrator", linestyle="--")
plt.plot(dt, error[2], label="EulerCromerIntegrator")
plt.xlabel("time step [s]")
plt.ylabel("Amplitude [J]")
plt.grid()
plt.legend()
plt.savefig("itegrator_timestep_plot.png")
plt.show()

# %%
start_time = 0
nsteps = 100_000
dt = 1

sys = SolarSystem(start_time, names, colors, masses, positions, velocities, accelerations)
integrator = VelocityVerletIntegrator(dt)
obs = Observables(sys, nsteps)
sim = Simulation(sys, integrator, steps=nsteps, obs=obs)
sim.run_simulation()

# Plot trajectories
plot_trajectories(obs)

# Plot energies
plot_sum_energies(obs)

# Calculate error
calculate_error(obs)