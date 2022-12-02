#%% Python molecular dynamics simulation of particles in 2 dimensions with real time animation
# BH, OF, MP, AJ, TS 2022-11-20, latest verson 2021-10-21

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# This local library contains the functions needed to perform force calculation
# Since this is by far the most expensive part of the code, it is 'wrapped aside'
# and accelerated using numba (https://numba.pydata.org/numba-doc/latest/user/5minguide.html)
import md_force_calculator as md

"""

    This script is rather long: sit back and try to understand its structure before jumping into coding.
    MD simulations are performed by a class (MDsimulator) that envelops both the parameters and the algorithm;
    in this way, performing several MD simulations can be easily done by just allocating more MDsimulator
    objects instead of changing global variables and/or writing duplicates.

    You are asked to implement two things:
    - Pair force and potential calculation (in md_force_calculator.py)
    - Temperature coupling (in md_template_numba.py)
    The latter is encapsulated into the class, so make sure you are modifying the variables and using the
    parameters of the class (the one you can access via 'self.variable_name' or 'self.function_name()').

"""
#%%
# Boltzmann constant
kB = 1.0

# Number of steps between heat capacity output
N_OUTPUT_HEAT_CAP = 1000

# You can use this global variable to define the number of steps between two applications of the thermostat
N_STEPS_THERMO = 10

# Lower (increase) this if the size of the disc is too large (small) when running run_animate()
DISK_SIZE = 750

#%%
class MDsimulator:

    """
        This class encapsulates the whole MD simulation algorithm
    """

    def __init__(self, 
        n = 48, 
        mass = 1.0, 
        numPerRow = 8, 
        initial_spacing = 1.12,
        T = 0.4, 
        dt = 0.01, 
        nsteps = 20000, 
        numStepsPerFrame = 100,
        startStepForAveraging = 100
        ):
        
        """
            This is the class 'constructor'; if you want to try different simulations with different parameters 
            (e.g. temperature, initial particle spacing) in the same scrip, allocate another simulator by passing 
            a different value as input argument. See the examples at the end of the script.
        """

        # Initialize simulation parameters and box
        self.n = n
        self.mass = 1.0
        self.invmass = 1.0/mass
        self.numPerRow = numPerRow
        self.Lx = numPerRow*initial_spacing
        self.Ly = numPerRow*initial_spacing
        self.area = self.Lx*self.Ly
        self.T = T
        self.kBT = kB*T
        self.dt = dt
        self.nsteps = nsteps
        self.numStepsPerFrame = numStepsPerFrame
        # Initialize positions, velocities and forces
        self.x = []
        self.y = []
        for i in range (n):
            self.x.append(self.Lx*0.95/numPerRow*((i % numPerRow) + 0.5*(i/numPerRow)))
            self.y.append(self.Lx*0.95/numPerRow*0.87*(i/numPerRow))
        
        # Numba likes numpy arrays much more than list
        # Numpy arrays are mutable, so can be passed 'by reference' to quick_force_calculation
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.vx = np.zeros(n, dtype=float)
        self.vy = np.zeros(n, dtype=float)
        self.fx = np.zeros(n, dtype=float)
        self.fy = np.zeros(n, dtype=float)

        # Initialize particles' velocity according to the initial temperature
        md.thermalize(self.vx, self.vy, np.sqrt(self.kBT/self.mass))
        # Initialize containers for energies
        self.sumEkin = 0
        self.sumEpot = 0
        self.sumEtot = 0
        self.sumEtot2 = 0
        self.sumVirial = 0
        self.outt = []
        self.ekinList = []
        self.epotList = []
        self.etotList = []
        self.startStepForAveraging = startStepForAveraging
        self.step = 0
        self.Epot = 0
        self.Ekin = 0
        self.Virial = 0
        self.Cv = 0
        self.Cv_list = []
        self.P = 0
        self.P_list = []

    def clear_energy_potential(self) :
        
        """
            Clear the temporary variables storing potential and kinetic energy
            Resets forces to zero
        """
        
        self.Epot = 0
        self.Ekin = 0
        self.Virial = 0
        for i in range(0, self.n):
            self.fx[i] = 0
            self.fy[i] = 0

    def update_forces(self) :

        """
            Updates forces and potential energy using functions
            pairEnergy and pairForce (which you coded above...)
        """
        
        tEpot, tVirial = md.quick_force_calculation(self.x, self.y, self.fx, self.fy, 
            self.Lx, self.Ly, self.n)
        self.Epot += tEpot
        self.Virial += tVirial
    
    def propagate(self) :

        """
            Performs an Hamiltonian propagation step and
            rescales velocities to match the input temperature 
            (THE LATTER YOU NEED TO IMPLEMENT!)
        """

        for i in range(0, self.n):
            self.vx, self.vy = md.thermalize(self.vx, self.vy, np.sqrt(self.kBT/self.mass))

            # Add the kinetic energy of particle i to the total
            self.Ekin += 0.5*self.mass*(self.vx[i]*self.vx[i] + self.vy[i]*self.vy[i])
            # Update the velocities with a half step
            self.vx[i] += self.fx[i]*self.invmass*0.5*self.dt
            self.vy[i] += self.fy[i]*self.invmass*0.5*self.dt
            # Update the coordinates
            self.x[i] += self.vx[i] * self.dt
            self.y[i] += self.vy[i] * self.dt
            # Apply p.c.b. and put particles back in the unit cell
            self.x[i] = self.x[i] % self.Lx
            self.y[i] = self.y[i] % self.Ly

        # for i in range(0, self.n):
        #     # At the first step we alread have the "full step" velocity
        #     if self.step > 0:
        #         # Update the velocities with a half step
        #         self.vx[i] += self.fx[i]*self.invmass*0.5*self.dt
        #         self.vy[i] += self.fy[i]*self.invmass*0.5*self.dt

        #     # Add the kinetic energy of particle i to the total
        #     self.Ekin += 0.5*self.mass*(self.vx[i]*self.vx[i] + self.vy[i]*self.vy[i])
        #     # Update the velocities with a half step
        #     self.vx[i] += self.fx[i]*self.invmass*0.5*self.dt
        #     self.vy[i] += self.fy[i]*self.invmass*0.5*self.dt
        #     # Update the coordinates
        #     self.x[i] += self.vx[i] * self.dt
        #     self.y[i] += self.vy[i] * self.dt
        #     # Apply p.c.b. and put particles back in the unit cell
        #     self.x[i] = self.x[i] % self.Lx
        #     self.y[i] = self.y[i] % self.Ly

    def md_step(self) :

        """
            Performs a full MD step
            (computes forces, updates positions/velocities)
        """

        # This function performs one MD integration step
        self.clear_energy_potential()
        self.update_forces()
        # Start averaging only after some initial spin-up time
        if self.step > self.startStepForAveraging:
            self.sumVirial += self.Virial
            self.sumEkin   += self.Ekin
            self.sumEpot   += self.Epot
            self.sumEtot   += self.Epot+self.Ekin
            self.sumEtot2  += (self.Epot+self.Ekin)*(self.Epot+self.Ekin)
        self.propagate()
        self.step += 1

    def integrate_some_steps(self, framenr=None) :

        """
            Performs MD steps in a prescribed time window
            Stores energies and heat capacity
        """

        for j in range(self.numStepsPerFrame) :
            self.md_step()
        t = self.step*self.dt
        self.outt.append(t)
        self.ekinList.append(self.Ekin)
        self.epotList.append(self.Epot)
        self.etotList.append(self.Epot + self.Ekin)
        self.Cv_list.append(self.Cv)
        self.P_list.append(self.P)
        if self.step >= self.startStepForAveraging and self.step % N_OUTPUT_HEAT_CAP == 0:
            EkinAv  = self.sumEkin/(self.step + 1 - self.startStepForAveraging)
            EtotAv = self.sumEtot/(self.step + 1 - self.startStepForAveraging)
            Etot2Av = self.sumEtot2/(self.step + 1 - self.startStepForAveraging)
            VirialAV = self.sumVirial/(self.step + 1 - self.startStepForAveraging)
            self.Cv = (Etot2Av - EtotAv * EtotAv) / (self.kBT * self.T)
            self.P = (2.0/self.area)*(EkinAv - VirialAV)
            # print('time', t, 'Cv =', self.Cv, 'P = ', self.P)

    def snapshot(self, framenr=None) :

        """
            This is an 'auxillary' function needed by animation.FuncAnimation
            in order to show the animation of the 2D Lennard-Jones system
        """

        self.integrate_some_steps(framenr)
        return self.ax.scatter(self.x, self.y, s=DISK_SIZE, marker='o', c="r"),

    def simulate(self) :

        """
            Performs the whole MD simulation
            If the total number of steps is not divisible by the frame size, then
            the simulation will undergo nsteps-(nsteps%numStepsPerFrame) steps
        """

        nn = self.nsteps//self.numStepsPerFrame
        # print("Integrating for "+str(nn*self.numStepsPerFrame)+" steps...")
        for i in range(nn) :
            self.integrate_some_steps()

    def simulate_animate(self) :

        """
            Performs the whole MD simulation, while producing and showing the
            animation of the molecular system
            CAREFUL! This will slow down the script execution considerably
        """

        self.fig = plt.figure()
        self.ax = plt.subplot(xlim=(0, self.Lx), ylim=(0, self.Ly))

        nn = self.nsteps//self.numStepsPerFrame
        # print("Integrating for "+str(nn*self.numStepsPerFrame)+" steps...") 
        self.anim = animation.FuncAnimation(self.fig, self.snapshot,
            frames=nn, interval=1000, blit=False, repeat=False)
        plt.axis('square')
        plt.show()  # show the animation
        # You may want to (un)comment the following 'waitforbuttonpress', depending on your environment
        # plt.waitforbuttonpress(timeout=20)

    def plot_energy(self, title="energies") :
        
        """
            Plots kinetic, potential and total energy over time
        """
        
        plt.figure()
        plt.title(title)
        plt.xlabel('time')
        plt.ylabel('energy')
        plt.plot(self.outt, self.ekinList, self.outt, self.epotList, self.outt, self.etotList, )
        plt.legend( ('Ekin','Epot','Etot') )
        plt.grid()
        plt.savefig(title + ".pdf")
        plt.show()

    def average_energy(self) :
        
        """
            Returns the average kinetic, potential and total energy
        """

        Etot_list = self.etotList[100:]
        sum_Etot = sum(Etot_list)
        return sum_Etot/len(Etot_list)

    def average_heat_capacity(self) :
        
        """
            Computes the average heat capacity
        """

        Cv_list = self.Cv_list[100:]
        sumCv = sum(Cv_list)
        return sumCv/len(Cv_list)

    def pressure(self) :
        
        """
            Computes the pressure
        """
        p_list = self.P_list[100:]
        sump = sum(p_list)
        return sump/len(p_list)


#%% Excerise 3.2 a)
def exercise_32a():
    T = 1.0
    dt = [0.01, 0.024, 0.0241, 0.02415, 0.0242, 0.0243]
    nsteps = 20_000
    for i in range(len(dt)):
        lj = MDsimulator(T=T, nsteps=nsteps, dt=dt[i])
        lj.simulate()
        lj.plot_energy(f"Energies as a function of time, dt = {dt[i]}")

if __name__ == "__main__" :
    #exercise_32a()
    pass

# %% Excerise 3.2 b)
def exercise_32b():
    T = 0.2
    dt = 0.01
    nsteps = 20_000
    lj = MDsimulator(T=T, nsteps=nsteps, dt=dt)
    lj.simulate()
    lj.plot_energy(f"Energies as a function of time, T = {T}")

if __name__ == "__main__" :
    #exercise_32b()
    pass

# %% Excerise 3.2 c)
def exercise_32c():
    T = [1, 0.2]
    nsteps = 10_000
    for i in range(len(T)):
        lj = MDsimulator(T=T[i], nsteps=nsteps, dt=0.01)
        lj.simulate_animate()
        lj.plot_energy(f"Energies as a function of time, T = {T[i]}, Andersen thermostat")

if __name__ == "__main__" :
    #exercise_32c()
    pass

# %% Excerise 3.2 d)
def exercise_32d():
    T = [temp for temp in np.arange(0.2, 1.2, 0.2)]
    nsteps = 100_000
    average_Cv = []
    average_Etot = []

    # calculate the average energy and heat capacity for each temperature 
    for i in range(len(T)):
        lj = MDsimulator(T=T[i], nsteps=nsteps, dt=0.01)
        lj.simulate()
        average_Cv.append(lj.average_heat_capacity())
        average_Etot.append(lj.average_energy())

    # plot the average energy for each temperature
    plt.figure()
    plt.title("Average energy as a function of temperature")
    plt.xlabel('temperature')
    plt.ylabel('energy')
    plt.plot(T, average_Etot)
    plt.legend( ('Etot') )
    plt.grid()
    plt.savefig("average_energy.pdf")
    plt.show()

    # plot the average heat capacity for each temperature
    plt.figure()
    plt.title("Average heat capacity as a function of temperature")
    plt.xlabel('heat capacity')
    plt.ylabel('energy')
    plt.plot(T, average_Cv)
    plt.legend( ('Cv') )
    plt.grid()
    plt.savefig("heat_capacity.pdf")
    plt.show()

if __name__ == "__main__" :
    #exercise_32d()
    pass

# %% Excerise 3.2 e)
def exercise_32e():
    T = [temp for temp in np.arange(0.2, 1.2, 0.2)]
    nsteps = 100_000
    initial_spacing = [1.12, 2.24]
    pressure = []

    # calculate the pressure for each temperature and initial spacing
    for i in range(len(initial_spacing)):
        pres = []
        for j in range(len(T)):
            lj = MDsimulator(T=T[j], nsteps=nsteps, dt=0.01, initial_spacing=initial_spacing[i])
            lj.simulate()
            pres.append(lj.pressure())
        pressure.append(pres)
    
    # plot the pressure for each temperature and initial spacing
    plt.figure()
    plt.title(f"Pressure as a function of temperature, initial spacing = {initial_spacing[0]}")
    plt.xlabel('temperature')
    plt.ylabel('pressure')
    plt.plot(T, pressure[0])
    plt.legend( ('initial spacing = 1.12') )
    plt.grid()
    plt.savefig("pressure1.pdf")
    plt.show()

    plt.figure()
    plt.title(f"Pressure as a function of temperature, initial spacing = {initial_spacing[1]}")
    plt.xlabel('temperature')
    plt.ylabel('pressure')
    plt.plot(T, pressure[1])
    plt.legend( ('initial spacing = 2.24') )
    plt.grid()
    plt.savefig("pressure2.pdf")
    plt.show()

if __name__ == "__main__" :
    exercise_32e()
