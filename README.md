# Electrostatic Landau-Damping Particle-In-Cell Simulation 

This is a github repository of Particle-In-Cell Python code for a 1D electrostatic plasma. It calculates the motion of electrons under the Poisson-Maxwell equation with an initial position perturbation that becomes a damped oscillation. 
<div>
    <p float = 'left'>
        <img src="/result/landau_phase_space1.gif"  width="50.5%">
        <img src="/result/landau_energy_loss1.png"  width="48.5%">
    </p>
</div
Ions in the plasma are assumed to be fixed while only electrons can move in the simulation. The gradient and Laplacian of the electric potential are approximated with the Finite Difference Method, then written in matrix form. Particle density is calcualted with numpy.bincount
- Time integration method - Leapfrog
- Interpolation method - Cloud-In-Cell
- Boundary Conditions - Periodic

### Parameters:
Some plasma parameters are normalized to order unity
- N = 100000 - Number of particles
- boxsize = 10 -  Physical space of simulation
- cells = 400 - Number of cells in the grid
- tEnd = 100 - Simulation total time
- n0 = 1 - Average electron density
- vth = 0.5 - Thermal velocity
- A = 0.15 - Strength of perturbation/oscillation
- k = 1 - Wavenumber of position oscillation
