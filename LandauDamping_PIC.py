#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: westonlarhette
"""

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks



######## Defining Parameters ########

N = 100000      # Number of particles
boxsize = 10    # Physical space of simulation
cells = 400     # Number of cells in the grid
t = 0           # Simulation start time
tEnd = 100      # Simulation end time
dt = 0.1        # Timestep
n0 = 1          # Average electron density
vth = 0.5       # Thermal velocity
A = 0.05        # Strength of perturbation/oscillation
k = 2          # Wavenumber of position oscillation



def getAcc( pos, cells, boxsize, n0, Grad_mtx, Lapl_mtx ):
        """ Calculates the acceleration on each particle due to electric field
        pos      is a Nx1 matrix of particle positions
        Grad_mtx     is a cells x cells matrix for calculating the gradient on the grid
        Lapl_mtx     is a cells x cells matrix for calculating the laplacian on the grid
        a        is an Nx1 matrix of accelerations """
        
        pos = np.mod(pos,boxsize)
        N = len(pos) # extract particle number from position vector
        
        dx = boxsize / cells # physical length of a cell
        
        # initialize Electron number density onto mesh by placing particles into the
        # nearest cells (j & j+1, with proper weights) and normalizing:
        j = np.floor(pos / dx).astype(int)
        jp1 = j + 1 # initializng j + 1 cells
        weight_j = (jp1 * dx - pos) / dx 
        weight_jp1 = (pos - j * dx) / dx

        jp1 = np.mod(jp1,cells) # ensures periodic boundary conditions - if
                                # jp1 gets to 400, this makes it zero instead


        # initializing particle density based off of meshgrid weights at j cells
        n = np.bincount(j[:,0], weights = weight_j[:,0], minlength = cells)
        # adding weights from j + 1 cells
        n += np.bincount(jp1[:,0], weights = weight_jp1[:,0], minlength = cells)
        n *= n0 * boxsize / ( N * dx)
        
        # solve Poisson's equation for electric potential: laplacian(phi) = n-n0
        phi_grid = spsolve(Lapl_mtx, n - n0, permc_spec="MMD_AT_PLUS_A")
        
        # apply derivative to get electric field
        E_grid = -Grad_mtx @ phi_grid # this gives electric field at each gridpoint/cell
        # below changes E_grid to calculate electric field at each particle
        E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]

        # acceleration
        a = -E
        return (a, E_grid)
    



######## Setting initial conditions ########

np.random.seed(42) # setting seed for random arrays

## Defining particle positions 
pos = np.random.rand(N,1) * boxsize

# Adding slight perturbation/oscillation to position
pos += A * np.sin(k * pos)

## Defining velocities - normally distributed around v = 0
vel = vth * np.random.randn(N,1)

# Constructing Grad_mtx to compute gradient (first derivative) of potential
dx = boxsize / cells
e = np.ones(cells)

diags = np.array([-1,1])

vals = np.vstack((-e,e))

Grad_mtx = sp.spdiags(vals,diags,cells,cells)
Grad_mtx = sp.lil_matrix(Grad_mtx)
Grad_mtx[0,cells-1] = -1
Grad_mtx[cells-1,0] = 1

Grad_mtx /= (2*dx)
Grad_mtx = sp.csr_matrix(Grad_mtx)

# Constructing Lapl_mtx to compute Laplacian (2nd derivative)

diags = np.array([-1,0,1])

vals = np.vstack((e,-2*e,e))

Lapl_mtx = sp.spdiags(vals, diags, cells, cells)
Lapl_mtx = sp.lil_matrix(Lapl_mtx) # transform mtx type to modify entries
Lapl_mtx[0,cells-1] = 1
Lapl_mtx[cells-1,0]  = 1

Lapl_mtx /= dx**2
Lapl_mtx = sp.csr_matrix(Lapl_mtx)



# calculate initial gravitational acceleration:
acc, E_grid = getAcc(pos, cells, boxsize, n0, Grad_mtx, Lapl_mtx)


# number of timesteps/frames
Nt = int(np.ceil(tEnd / dt))


#%%


######## Simulation Start ########


# Initializing data of energy stored in electric field
E_energyhistory = []
E_energy = np.sum(E_grid**2) * dx
E_energyhistory.append(E_energy)

# Initiazing data of electric field value at each cell
E_gridarray = np.zeros((4,cells))
E_gridarray[0,:] = E_grid


## Time loop to collect electric field data ##
# Leapfrog Method
for i in range(Nt):
    # (1/2) kick
    vel += acc * dt / 2.0
    
    # drift (and apply periodic boundary conditions)
    pos += vel * dt
    pos = np.mod(pos, boxsize)
    
    # update acceleration & electric field data
    acc, E_grid = getAcc(pos, cells, boxsize, n0, Grad_mtx, Lapl_mtx)
    E_energy = np.sum(E_grid**2) * dx
    E_energyhistory.append(E_energy)
    
    if i == Nt//50:
        E_gridarray[1,:] = E_grid
    if i == Nt//2:
        E_gridarray[2,:] = E_grid
    if i == (Nt-1):
        E_gridarray[3,:] = E_grid

    # (1/2) kick
    vel += acc * dt / 2.0
    
    # update time
    t += dt



# Resetting variables for second time loop
t = 0
pos = np.random.rand(N,1) * boxsize
pos += A * np.sin(k * pos)
vel = vth * np.random.randn(N,1)
acc = getAcc(pos, cells, boxsize, n0, Grad_mtx, Lapl_mtx)[0]


## Time loop for animating phase space ##
figure, ax = plt.subplots(figsize = (7,5), dpi=80)
def update(frame):
    global vel, pos, acc, t
    
    vel += acc * dt / 2.0
    
    pos += vel * dt
    pos = np.mod(pos, boxsize)
    
    acc = getAcc(pos, cells, boxsize, n0, Grad_mtx, Lapl_mtx)[0]
    
    vel += acc * dt / 2.0
    
    t += dt
    
    # Plot phase space
    ax.clear()
    ax.scatter(pos, vel, s = 0.4, color='black', alpha = 0.3)
    ax.set_xlim([0,boxsize])
    ax.set_ylim([-2, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title(fr'Landau Damping in Phase Space: $k = {k}, A = {A}$')
    ax.text(0.02, 0.98, f't = {t:.2f}', 
            transform=ax.transAxes,
            fontsize = 10,
            verticalalignment='top',
            bbox=dict(facecolor='green', boxstyle='square', alpha = 0.3))
    
landau_ani = FuncAnimation(figure, update, frames = Nt, interval = 50)
plt.show()


#%%
# Switch to inline plotting
matplotlib.use('module://ipykernel.pylab.backend_inline')


##### Finding peaks of damped signal #####


E_energyhistory = np.array(E_energyhistory)

# Defining time array for plotting energy signal
time_array = np.arange(len(E_energyhistory)) * dt

peaks, _ = find_peaks(E_energyhistory[time_array < 30])

# Plotting peaks and raw signal
plt.figure()
plt.semilogy(time_array[peaks], E_energyhistory[peaks], 'o--', label='Peak envelope')
plt.semilogy(time_array, E_energyhistory, alpha=0.3, label='Full signal')
plt.xlim(-1,31)
plt.xlabel('Simulation Time')
plt.ylabel('Energy stored in electric field (~$E^2$)')
plt.legend()
plt.title(fr'Energy Loss of Oscillation due to Landau Damping: $k = {k}, A = {A}$')
plt.grid(True)
plt.savefig('Landau Damping')
plt.show()

#%%
matplotlib.use('module://ipykernel.pylab.backend_inline')


##### Plotting E(x) snapshots at each cell for different times #####

# Defining x-axis array of cells
x_axis_cells = np.arange(400) 

plt.figure()
plt.plot(x_axis_cells, E_gridarray[0,:], 'black', label = 't = 0')
plt.plot(x_axis_cells, E_gridarray[1,:], 'blue', label = 't = T/50')
plt.plot(x_axis_cells, E_gridarray[2,:], 'red', label = 't = T/2')
plt.plot(x_axis_cells, E_gridarray[3,:], 'green', label = 't = T')
plt.grid(True)
plt.title(fr'E(x) at different time snapshots: $k = {k}, A = {A}$')
plt.xlabel('x (cell number)')
plt.ylabel('E')
plt.xlim([0,400])
plt.legend()
plt.show()



