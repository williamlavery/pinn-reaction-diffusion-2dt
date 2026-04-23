"""Finite-difference PDE utilities used for forward 2D BINN simulations.

This module assembles diffusion operators and integrates reaction-diffusion
systems on regular grids for model evaluation.
"""

import os
import time
import itertools
import numpy as np
import scipy.io as sio
import scipy.optimize
from scipy import integrate
from scipy import sparse
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from IPython.display import clear_output

def Du_2d(D_flat, dx, dy, nx, ny):
    """Assemble the sparse operator for $\nabla\cdot(D\nabla u)$ on a 2D grid.

    This function generates a finite-difference approximation matrix using central 
    differences, enforcing no-flux (mirrored) boundary conditions at the edges.

    Args:
        D_flat (np.ndarray): Flattened diffusion field of length `nx * ny`.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        nx (int): Number of grid cells in the x-direction.
        ny (int): Number of grid cells in the y-direction.

    Returns:
        scipy.sparse.coo_matrix: A sparse matrix `A` of shape `(nx*ny, nx*ny)` 
        such that `A @ u_flat` approximates $\nabla\cdot(D\nabla u)$.
    """

    size = nx * ny

    D = D_flat.reshape((nx, ny))

    # Precompute constants
    inv_dx2 = 1 / (2 * dx ** 2)
    inv_dy2 = 1 / (2 * dy ** 2)

    # X-direction neighbors (left and right)
    D_left = np.roll(D, shift=1, axis=0)  # Left neighbor (i-1)
    D_right = np.roll(D, shift=-1, axis=0)  # Right neighbor (i+1)
    
    # Handle boundary conditions in x-direction
    D_left[0, :] = D[1, :]  # Enforce BC at left edge
    D_right[-1, :] = D[-2, :]  # Enforce BC at right edge

    # Y-direction neighbors (up and down)
    D_up = np.roll(D, shift=1, axis=1)  # Upper neighbor (j-1)
    D_down = np.roll(D, shift=-1, axis=1)  # Lower neighbor (j+1)

    # Handle boundary conditions in y-direction
    D_up[:, 0] = D[:, 1]  # Enforce BC at top edge
    D_down[:, -1] = D[:, -2]  # Enforce BC at bottom edge

    # X-direction contributions
    data_x_left = (D_left + D) * inv_dx2
    data_x_center = -(2 * D + D_left + D_right) * inv_dx2
    data_x_right = (D + D_right) * inv_dx2

    # Y-direction contributions
    data_y_up = (D_up + D) * inv_dy2
    data_y_center = -(2 * D + D_up + D_down) * inv_dy2
    data_y_down = (D + D_down) * inv_dy2

    # Combine data for x and y directions
    data = np.dstack((data_x_left, data_x_center, data_x_right,
        data_y_up, data_y_center, data_y_down))
    data = np.swapaxes(data, 0,1).flatten()
        
    # Construct row and column indices
    indices = np.arange(nx * ny).reshape(nx, ny)
    
    # X-direction indices
    row_x = np.repeat(indices, 3)  # Replicate for left, center, right
    row_y =  np.repeat(indices, 3) # Replicate for up, center, down
    row = np.dstack((row_x,row_y)).flatten()
    
    col_x_left = np.where(indices % nx > 0, indices - 1, indices + 1)
    col_x_right = np.where(indices % nx < nx - 1, indices + 1, indices - 1)
    col_y_upper = np.where(indices // nx > 0, indices - nx, indices + nx)
    col_y_lower = np.where(indices // nx < ny - 1, indices + nx, indices - nx)
    
    col = np.dstack((
        col_x_left,
        indices,
        col_x_right,
        col_y_upper,
        indices,
        col_y_lower,
    )).flatten()

    return sparse.coo_matrix((data, (row, col)), shape=(size, size))


def PDE_RHS_2D(t, y, x1, x2, D, f):
    """Compute the RHS for a 2D diffusion-reaction system on a flattened grid.

    Constructs the spatial derivatives to evaluate $u_t = \nabla\cdot(D(u)\nabla u) + u f(u)$.
    Currently, this implementation only evaluates autonomous functions where D and f 
    depend exclusively on the state vector y, not time t.

    Args:
        t (float): Current simulation time (unused in RHS mapping, required for ODE solvers).
        y (np.ndarray): Flattened state vector of the system.
        x1 (np.ndarray): 1D array of x-coordinates.
        x2 (np.ndarray): 1D array of y-coordinates.
        D (callable): Function returning diffusion coefficients given state `y`.
        f (callable): Function returning growth/reaction rates given state `y`.

    Returns:
        np.ndarray: Evaluated rate of change (RHS) of the identical shape as `y`.
    """
    dx1 = x1[1] - x1[0]
    dx2 = x2[1] - x2[0]
    nx1, nx2 = len(x1), len(x2)
    try:
        # Case 4: D(y) and f(y) (least variable case)
        Du_mat = Du_2d(D(y), dx1, dx2, nx1, nx2)
        return Du_mat.dot(y) + y * f(y)
    except Exception as e4:
        raise RuntimeError(
            "Ensure D and f are correctly defined "
            "for the expected input combinations (y). Original error: " + str(e4)
        )
        

def PDE_sim_old_2d(RHS, IC, x1, x2, t, D, f, numtsim=1000, numxsim1=200, numxsim2=200, clear=True):
    """Run forward simulation and sample outputs back on the original grid.

    Interpolates the provided Initial Condition onto a higher-resolution internal 
    simulation grid, integrates the PDE using the 'dopri5' solver, and then interpolates 
    the resulting states back to the original observation points for specified times `t`.

    Args:
        RHS (callable): Function computing the RHS of the PDE, generally `PDE_RHS_2D`.
        IC (np.ndarray): Initial condition matrix of shape `(len(x1), len(x2))`.
        x1 (np.ndarray): Original observation grid coordinates in x.
        x2 (np.ndarray): Original observation grid coordinates in y.
        t (np.ndarray): Array of observation times to return states for.
        D (callable): Diffusion function mapped onto the system state.
        f (callable): Growth/Reaction function mapped onto the system state.
        numtsim (int, optional): Number of internal integration time steps. Defaults to 1000.
        numxsim1 (int, optional): Internal simulation grid resolution in x. Defaults to 200.
        numxsim2 (int, optional): Internal simulation grid resolution in y. Defaults to 200.
        clear (bool, optional): Whether to clear notebook output to suppress print accumulation. Defaults to True.

    Returns:
        np.ndarray: Simulated system states mapped back to the original grid shape `(len(x1), len(x2), len(t))`.
    """

    def initialize_simulation(t, x1, x2, IC, numtsim, numxsim1, numxsim2):
        """Initialize simulation grids and interpolate initial conditions."""
        t_sim = np.linspace(np.min(t), np.max(t), numtsim)
        x1_sim = np.linspace(np.min(x1), np.max(x1), numxsim1)
        x2_sim = np.linspace(np.min(x2), np.max(x2), numxsim2)

        f_interpolate = RegularGridInterpolator((x1, x2), IC)
        #print("Interpolating initial conditions...")
        queries_sim = (np.array(list(itertools.product(x1_sim, x2_sim))))
        y0 = f_interpolate(queries_sim).flatten()
        return t_sim, x1_sim, x2_sim, y0

    def find_write_indices(t, t_sim):
        """Find time indices for writing output."""
        return np.array([np.abs(tp - t_sim).argmin() for tp in t])

    def integrate(t_sim, y0, RHS_func, write_indices, query_grid, solver_shape):
        """Perform integration using the provided RHS function."""
        y = np.zeros(solver_shape)
        y[..., 0] = IC
        
        r = scipy.integrate.ode(RHS_func).set_integrator("dopri5").set_initial_value(y0, t[0])

        write_count = 0
        
        for i in range(1, len(t_sim)):
            if i in write_indices:
                
                write_count += 1
                sol = r.integrate(t_sim[i]).reshape(query_grid[1])
                f_interpolate = RegularGridInterpolator(query_grid[0], sol)
                y[..., write_count] = f_interpolate(query_grid[2]).reshape(IC.shape)
                
                if clear:
                    clear_output() # clear_output used for removing error messages dispalyed to screen
            else:
                r.integrate(t_sim[i])
            if not r.successful():
                print("Integration failed")
                return 1e6 * np.ones_like(y)
            
            # Print progress
            progress = (i + 1) / len(t_sim) * 100
            print("\rProgress: {}% complete".format(round(progress,2)), end="")
        print("\rProgress: 100.00% complete")
        return y

    # Main simulation logic
    
    t_sim, x1_sim, x2_sim, y0 = initialize_simulation(t, x1, x2, IC, numtsim, numxsim1, numxsim2)
    write_indices = find_write_indices(t, t_sim)

    def RHS_2D(t, y):
        return RHS(t, y.flatten(), x1_sim, x2_sim, D, f)

    solver_shape = (len(x1), len(x2), len(t))
    query_grid = ((x1_sim, x2_sim), (numxsim1, numxsim2), np.array(list(itertools.product(x1, x2))))
    
    return integrate(t_sim, y0, RHS_2D, write_indices, query_grid, solver_shape)