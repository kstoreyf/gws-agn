"""
Plotting utilities for GW-AGN analysis.

This module provides functions for:
- Sky plots (RA/Dec visualization)
- Likelihood grid plotting
- MCMC posterior plotting
"""

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import h5py
import yaml
from scipy.interpolate import interp1d
from scipy.special import logsumexp

# Import local modules
import utils
import run_inference


# Function to create a nice sky plot from RA and Dec coordinates
def create_skyplot_single(ra, dec, figsize=(10, 6), projection='mollweide', 
                   colormap='viridis', alpha=0.05, s=0.1, title=None, 
                   gridlines=True, grid_color='gray', grid_alpha=0.3,
                   background_color='white', c='k'):
    """
    Create a beautiful sky plot of astronomical coordinates.
    
    Parameters:
    -----------
    ra : array-like
        Right Ascension in degrees
    dec : array-like
        Declination in degrees
    figsize : tuple, optional
        Figure size in inches (width, height)
    projection : str, optional
        Map projection type ('mollweide', 'hammer', 'aitoff', etc.)
    colormap : str, optional
        Matplotlib colormap name
    alpha : float, optional
        Transparency of points
    s : float, optional
        Point size
    title : str, optional
        Plot title
    gridlines : bool, optional
        Whether to show grid lines
    grid_color : str, optional
        Color of grid lines
    grid_alpha : float, optional
        Transparency of grid lines
    background_color : str, optional
        Background color of the plot
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Convert RA from degrees to radians for plotting
    # In astronomy, RA increases eastward, but in math, angle increases counterclockwise
    # So we need to convert RA to mathematical angle (theta) by:
    # theta = 2*pi - RA*pi/180 (convert degrees to radians and invert direction)
    theta = np.radians(ra)
    # Convert to range expected by projection (-pi to pi)
    theta = np.pi - theta  # Flip RA so east is to the left (astronomical convention)
    
    # Convert Dec to radians
    phi = np.radians(dec)
    
    # Create figure with specified projection
    fig = plt.figure(figsize=figsize, facecolor=background_color)
    ax = fig.add_subplot(111, projection=projection, facecolor=background_color)
    
    # Plot points
    c = ax.scatter(theta, phi, c=c,
                  alpha=alpha, s=s, edgecolors='none')
    
    
    # Configure grid
    if gridlines:
        ax.grid(True, color=grid_color, alpha=grid_alpha, linestyle='--')
    
    # Configure tick labels
    # For RA, create labels at 30-degree intervals (2 hours)
    ra_labels = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
    ra_ticks = np.pi - np.radians(ra_labels)
    ax.set_xticks(ra_ticks)
    ra_labels_str = [f'{h:d}h' for h in (ra_labels // 15)]
    ax.set_xticklabels(ra_labels_str)
    
    # For Dec, create labels at 30-degree intervals from -90 to +90
    dec_labels = np.array([-90, -60, -30, 0, 30, 60, 90])
    dec_ticks = np.radians(dec_labels)
    ax.set_yticks(dec_ticks)
    ax.set_yticklabels([f'{d:+d}°' for d in dec_labels])
    
    # Set tick label colors
    ax.tick_params(axis='both')#, colors='white')
    
    # Remove frame
    ax.spines['geo'].set_visible(False)
    
    return fig, ax




# Function to create a nice sky plot from RA and Dec coordinates
def create_skyplot(ra_arr, dec_arr, figsize=(10, 6), projection='mollweide', 
                   colormap='viridis', alpha_arr=None, s_arr=None, marker_arr=None, 
                   label_arr=None,
                   title=None, 
                   gridlines=True, grid_color='gray', grid_alpha=0.3,
                   background_color='white', c_arr='k'):
    """
    Create a beautiful sky plot of astronomical coordinates.
    
    Parameters:
    -----------
    ra : array-like
        Right Ascension in degrees
    dec : array-like
        Declination in degrees
    figsize : tuple, optional
        Figure size in inches (width, height)
    projection : str, optional
        Map projection type ('mollweide', 'hammer', 'aitoff', etc.)
    colormap : str, optional
        Matplotlib colormap name
    alpha : float, optional
        Transparency of points
    s : float, optional
        Point size
    title : str, optional
        Plot title
    gridlines : bool, optional
        Whether to show grid lines
    grid_color : str, optional
        Color of grid lines
    grid_alpha : float, optional
        Transparency of grid lines
    background_color : str, optional
        Background color of the plot
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
        
    # Create figure with specified projection
    fig = plt.figure(figsize=figsize, facecolor=background_color)
    ax = fig.add_subplot(111, projection=projection, facecolor=background_color)
    
    if marker_arr is None:
        marker_arr = ['o' for i in range(len(ra_arr))]
        
    if alpha_arr is None:
        alpha_arr = [0.05 for i in range(len(ra_arr))]
        
    if s_arr is None:
        s_arr = [0.1 for i in range(len(ra_arr))]
    
    #for ra, dec, alpha, s, c in zip(ra_arr, dec_arr, alpha_arr, s_arr, c_arr):
    for i in range(len(ra_arr)):
        
        ra, dec, alpha, s, c, marker = ra_arr[i], dec_arr[i], alpha_arr[i], s_arr[i], c_arr[i], marker_arr[i]
        # Convert RA from degrees to radians for plotting
        # In astronomy, RA increases eastward, but in math, angle increases counterclockwise
        # So we need to convert RA to mathematical angle (theta) by:
        # theta = 2*pi - RA*pi/180 (convert degrees to radians and invert direction)
        theta = np.radians(ra)
        # Convert to range expected by projection (-pi to pi)
        theta = np.pi - theta  # Flip RA so east is to the left (astronomical convention)
        
        # Convert Dec to radians
        phi = np.radians(dec)

        if label_arr is not None:
            label = label_arr[i]
        else:
            label = None
        # Plot points
        c = ax.scatter(theta, phi, c=c,
                    alpha=alpha, s=s, marker=marker, edgecolors='none', label=label)
    
    
    # Configure grid
    if gridlines:
        ax.grid(True, color=grid_color, alpha=grid_alpha, linestyle='--')
    
    # Configure tick labels
    # For RA, create labels at 30-degree intervals (2 hours)
    ra_labels = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
    ra_ticks = np.pi - np.radians(ra_labels)
    ax.set_xticks(ra_ticks)
    ra_labels_str = [f'{h:d}h' for h in (ra_labels // 15)]
    ax.set_xticklabels(ra_labels_str)
    
    # For Dec, create labels at 30-degree intervals from -90 to +90
    dec_labels = np.array([-90, -60, -30, 0, 30, 60, 90])
    dec_ticks = np.radians(dec_labels)
    ax.set_yticks(dec_ticks)
    ax.set_yticklabels([f'{d:+d}°' for d in dec_labels])
    
    # Set tick label colors
    ax.tick_params(axis='both')#, colors='white')
    
    # Remove frame
    ax.spines['geo'].set_visible(False)
    
    plt.legend(loc='upper right', fontsize='small')
    
    return fig, ax


# ============================================================================
# Likelihood plotting functions
# ============================================================================

def get_likelihood_data(fn_config_inference):
    """
    Load likelihood data from a config inference file.
    
    Parameters:
    -----------
    fn_config_inference : str
        Path to the inference config YAML file
        
    Returns:
    --------
    dict : Dictionary containing:
        - log_likelihood_grid: 2D array of log likelihood values
        - H0_grid: 1D array of H0 values
        - alpha_agn_grid: 1D array of alpha_agn values
        - truth_H0: True H0 value from config
        - truth_alpha_agn: True alpha_agn value computed from config
        - config_inference: The loaded inference config dict
        - config_data: The loaded data config dict
    """
    print(fn_config_inference)
    
    with open(fn_config_inference, 'r') as f:
        config_inference = yaml.safe_load(f)
    
    config_data_path = config_inference['fn_config_data']
    print(config_data_path)
    
    with open(config_data_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Get output file from config
    fn_inf = config_inference['paths']['fn_inf']
        
    # Load likelihood grid
    likelihood_results = run_inference.load_likelihood_grid(fn_inf)
    
    # Extract data
    log_likelihood_grid = likelihood_results['log_likelihood_grid']
    H0_grid = likelihood_results['H0_grid']
    alpha_agn_grid = likelihood_results['alpha_agn_grid']
    
    print(f"Loaded likelihood grid with shape: {log_likelihood_grid.shape}")
    print(f"H0 grid: {len(H0_grid)} points from {H0_grid.min():.1f} to {H0_grid.max():.1f}")
    print(f"alpha_agn grid: {len(alpha_agn_grid)} points from {alpha_agn_grid.min():.3f} to {alpha_agn_grid.max():.3f}")
    
    # Get truth values from config
    truth_H0 = config_data['cosmology']['H0']
    truth_f_agn = config_data['gw_injection']['f_agn']
    truth_lambda_agn = config_data['gw_injection']['lambda_agn']
    
    with h5py.File(f'{config_data["paths"]["dir_mock"]}/{config_data["paths"]["name_cat"]}', 'r') as f:
        N_gal_truth = f.attrs['n_gal']
        N_agn_truth = f.attrs['n_agn']
    
    print(f"N_gal_truth: {N_gal_truth}, N_agn_truth: {N_agn_truth}"
          f"truth_f_agn: {truth_f_agn}, truth_lambda_agn: {truth_lambda_agn}")
    _, truth_alpha_agn = utils.compute_gw_host_fractions(N_gal_truth, N_agn_truth, truth_f_agn, lambda_agn=truth_lambda_agn)
    print(f"truth_alpha_agn: {truth_alpha_agn}")
    return {
        'log_likelihood_grid': log_likelihood_grid,
        'H0_grid': H0_grid,
        'alpha_agn_grid': alpha_agn_grid,
        'truth_H0': truth_H0,
        'truth_alpha_agn': truth_alpha_agn,
        'truth_f_agn': truth_f_agn,
        'truth_lambda_agn': truth_lambda_agn,
        'N_gal_truth': N_gal_truth,
        'N_agn_truth': N_agn_truth,
        'config_inference': config_inference,
        'config_data': config_data
    }


def plot_likelihood_data(likelihood_data, figsize=(7, 6)):
    """
    Plot likelihood data as a contour plot with marginalized distributions.
    
    Parameters:
    -----------
    likelihood_data : dict
        Dictionary returned by get_likelihood_data()
    figsize : tuple, optional
        Figure size (default: (7, 6))
    """
    log_likelihood_grid = likelihood_data['log_likelihood_grid']
    H0_grid = likelihood_data['H0_grid']
    alpha_agn_grid = likelihood_data['alpha_agn_grid']
    truth_H0 = likelihood_data['truth_H0']
    truth_alpha_agn = likelihood_data['truth_alpha_agn']
    
    # Prepare the likelihood grid
    # Convert to likelihood (relative, normalized) or use log directly
    log_likelihood_max = np.nanmax(log_likelihood_grid)
    likelihood_grid = np.exp(log_likelihood_grid - log_likelihood_max)
    
    # Ensure grid is in correct shape: (n_H0, n_alpha_agn)
    # The grid from compute_likelihood_grid is (n_H0, n_alpha_agn)
    # where likelihood_grid[i, j] corresponds to H0_grid[i], alpha_agn_grid[j]
    print(f"Original grid shape: {likelihood_grid.shape}")
    print(f"H0_grid length: {len(H0_grid)}, alpha_agn_grid length: {len(alpha_agn_grid)}")
    
    # if likelihood_grid.shape[0] == len(alpha_agn_grid) and likelihood_grid.shape[1] == len(H0_grid):
    #     # Grid was transposed, transpose back to (n_H0, n_alpha_agn)
    #     print("Transposing grid from (n_alpha_agn, n_H0) to (n_H0, n_alpha_agn)")
    #     likelihood_grid = likelihood_grid.T
    
    print(f"Working grid shape: {likelihood_grid.shape} (should be (n_H0={len(H0_grid)}, n_alpha_agn={len(alpha_agn_grid)}))")
    
    # Compute marginalized likelihoods
    # Marginalize over alpha_agn (axis=1) to get P(H0)
    likelihood_H0 = np.trapz(likelihood_grid, alpha_agn_grid, axis=1)
    likelihood_H0 = likelihood_H0 / np.trapz(likelihood_H0, H0_grid)  # Normalize
    
    # Marginalize over H0 (axis=0) to get P(alpha_agn)
    likelihood_alpha_agn = np.trapz(likelihood_grid, H0_grid, axis=0)
    likelihood_alpha_agn = likelihood_alpha_agn / np.trapz(likelihood_alpha_agn, alpha_agn_grid)  # Normalize
    
    # Create figure with subplots: 2x3 grid to make room for colorbar
    # Top-left: marginalized H0
    # Top-middle: empty (or title)
    # Top-right: empty (for colorbar space)
    # Bottom-left: 2D contour
    # Bottom-middle: marginalized alpha_agn
    # Bottom-right: colorbar
    fig = plt.figure(figsize=(figsize[0] * 1.2, figsize[1]))
    gs = fig.add_gridspec(2, 3, width_ratios=[3, 1, 0.15], height_ratios=[1, 3], 
                          hspace=0.1, wspace=0.1)
    
    # Create meshgrid for plotting
    # For contourf, we need meshgrid where first dimension is alpha_agn, second is H0
    # meshgrid(H0_grid, alpha_agn_grid) creates: H0_mesh[i,j] = H0_grid[j], alpha_agn_mesh[i,j] = alpha_agn_grid[i]
    # This gives shape (len(alpha_agn_grid), len(H0_grid))
    H0_mesh, alpha_agn_mesh = np.meshgrid(H0_grid, alpha_agn_grid, indexing='xy')
    
    # Apply transpose for plotting: likelihood_grid is (n_H0, n_alpha_agn)
    # After transpose: (n_alpha_agn, n_H0) which matches meshgrid shape
    likelihood_grid_plot = likelihood_grid.T
    #likelihood_grid_plot = likelihood_grid
    
    # Bottom-left: 2D contour plot
    ax_main = fig.add_subplot(gs[1, 0])
    im = ax_main.contourf(H0_mesh, alpha_agn_mesh, likelihood_grid_plot, levels=50, cmap='viridis', alpha=0.7)
    
    # Add green truth lines
    ax_main.axvline(truth_H0, color='green', linestyle='-', linewidth=1, label=f'Truth H0: {truth_H0:.1f}')
    ax_main.axhline(truth_alpha_agn, color='green', linestyle='-', linewidth=1, label=f'Truth α_AGN: {truth_alpha_agn:.3f}')
    
    ax_main.set_xlabel(r'$H_0$ [km/s/Mpc]', fontsize=14)
    ax_main.set_ylabel(r'$\alpha_{AGN}$', fontsize=14)
    ax_main.legend(fontsize=12)
    ax_main.grid(True, alpha=0.3)
    
    # Top-left: marginalized H0 distribution
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.plot(H0_grid, likelihood_H0, 'k-', linewidth=2)
    ax_top.axvline(truth_H0, color='green', linestyle='-', linewidth=1)
    ax_top.set_ylabel('P(H₀)', fontsize=12)
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_xticks([])
    ax_top.grid(True, alpha=0.3)
    
    # Bottom-middle: marginalized alpha_agn distribution
    ax_right = fig.add_subplot(gs[1, 1])
    ax_right.plot(likelihood_alpha_agn, alpha_agn_grid, 'k-', linewidth=2)
    ax_right.axhline(truth_alpha_agn, color='green', linestyle='-', linewidth=1)
    ax_right.set_xlabel('P(α)', fontsize=12)
    ax_right.set_ylim(ax_main.get_ylim())
    ax_right.set_yticks([])
    ax_right.grid(True, alpha=0.3)
    
    # Top-middle: empty (or could add title)
    ax_empty = fig.add_subplot(gs[0, 1])
    ax_empty.axis('off')
    #ax_empty.text(0.5, 0.5, 'Likelihood Grid', fontsize=16, ha='center', va='center', transform=ax_empty.transAxes)
    
    # Top-right: empty
    ax_empty2 = fig.add_subplot(gs[0, 2])
    ax_empty2.axis('off')
    
    # Add colorbar on the far right
    cbar_ax = fig.add_subplot(gs[1, 2])
    cbar = plt.colorbar(im, cax=cbar_ax, label='Relative Likelihood')
    
    plt.tight_layout()
    plt.show()


def plot_likelihood_fagn_lambdaagn(likelihood_data, n_fagn=50, n_lambda=50, 
                                    figsize_2d=(7, 5), figsize_1d=(16, 6), 
                                    levels=20, cmap='viridis'):
    """
    Plot 2D likelihood and marginalized distributions for f_agn and lambda_agn
    from a likelihood grid on (H0, alpha_agn).
    
    This function converts the likelihood grid on (H0, alpha_agn) to a likelihood
    on (f_agn, lambda_agn) by:
    1. Marginalizing over H0 to get 1D likelihood on alpha_agn
    2. Creating a grid of (f_agn, lambda_agn) values
    3. For each pair, computing the corresponding alpha_agn and getting the likelihood
    4. Normalizing and marginalizing to get 1D distributions
    
    Parameters:
    -----------
    likelihood_data : dict
        Dictionary returned by get_likelihood_data()
    n_fagn : int, optional
        Number of grid points for f_agn (default: 50)
    n_lambda : int, optional
        Number of grid points for lambda_agn (default: 50)
    figsize_2d : tuple, optional
        Figure size for 2D plot (default: (10, 8))
    figsize_1d : tuple, optional
        Figure size for 1D plots (default: (16, 6))
    levels : int, optional
        Number of contour levels (default: 20)
    cmap : str, optional
        Colormap name (default: 'viridis')
    """
    # Extract data from likelihood_data
    log_likelihood_grid = likelihood_data['log_likelihood_grid']
    H0_grid = likelihood_data['H0_grid']
    alpha_agn_grid = likelihood_data['alpha_agn_grid']
    truth_f_agn = likelihood_data.get('truth_f_agn')
    truth_lambda_agn = likelihood_data.get('truth_lambda_agn')
    N_gal = likelihood_data.get('N_gal_truth')
    N_agn = likelihood_data.get('N_agn_truth')
    config_data = likelihood_data.get('config_data')
    
    # Compute a simple checksum of the input likelihood to verify it's being used
    likelihood_checksum = np.sum(log_likelihood_grid) + np.sum(H0_grid) + np.sum(alpha_agn_grid)
    print(f"Input likelihood checksum: {likelihood_checksum:.6f} (use this to verify different inputs)")
    
    # Get N_gal and N_agn if not in likelihood_data
    if N_gal is None or N_agn is None:
        if config_data is None:
            raise ValueError("N_gal or N_agn missing from likelihood_data and config_data not available")
        # Load from catalog file
        fn_cat = f'{config_data["paths"]["dir_mock"]}/{config_data["paths"]["name_cat"]}'
        with h5py.File(fn_cat, 'r') as f:
            N_gal = int(f.attrs['n_gal'])
            N_agn = int(f.attrs['n_agn'])
    
    print(f"N_gal = {N_gal}, N_agn = {N_agn}")
    
    # Step 1: Marginalize over H0 to get 1D likelihood on alpha_agn
    # The grid shape is (n_H0, n_alpha_agn) - we need to marginalize over H0 (axis 0)
    # Use logsumexp for numerical stability
    log_likelihood_grid_work = log_likelihood_grid.copy()
    H0_grid_work = H0_grid.copy()
    alpha_agn_grid_work = alpha_agn_grid.copy()
    
    # Debug: print input grid info
    print(f"Input log_likelihood_grid shape: {log_likelihood_grid.shape}")
    print(f"Input log_likelihood_grid range: [{log_likelihood_grid.min():.2f}, {log_likelihood_grid.max():.2f}]")
    print(f"Input H0_grid: {len(H0_grid)} points from {H0_grid.min():.1f} to {H0_grid.max():.1f}")
    print(f"Input alpha_agn_grid: {len(alpha_agn_grid)} points from {alpha_agn_grid.min():.4f} to {alpha_agn_grid.max():.4f}")
    
    # Ensure grid is in correct shape: (n_H0, n_alpha_agn)
    # Check if grid was transposed for plotting (shape would be n_alpha_agn, n_H0)
    #if log_likelihood_grid_work.shape[0] == len(alpha_agn_grid_work) and log_likelihood_grid_work.shape[1] == len(H0_grid_work):
    #    # Grid was transposed for plotting, transpose back
    #    print("Transposing grid from (n_alpha_agn, n_H0) to (n_H0, n_alpha_agn)")
    #    log_likelihood_grid_work = log_likelihood_grid_work.T
    
    print(f"Working grid shape: {log_likelihood_grid_work.shape} (should be (n_H0, n_alpha_agn))")
    
    # Marginalize over H0: log P(alpha_agn) = log sum_H0 exp(log P(H0, alpha_agn))
    log_likelihood_alpha_agn = logsumexp(log_likelihood_grid_work, axis=0)  # Shape: (n_alpha_agn,)
    
    # Debug: print marginalized likelihood info
    print(f"Marginalized log_likelihood_alpha_agn range: [{log_likelihood_alpha_agn.min():.2f}, {log_likelihood_alpha_agn.max():.2f}]")
    print(f"Marginalized log_likelihood_alpha_agn argmax: alpha_agn = {alpha_agn_grid_work[np.argmax(log_likelihood_alpha_agn)]:.4f}")
    
    # Convert to probability (normalize)
    likelihood_alpha_agn = np.exp(log_likelihood_alpha_agn - log_likelihood_alpha_agn.max())  # Subtract max for numerical stability
    likelihood_alpha_agn = likelihood_alpha_agn / np.trapz(likelihood_alpha_agn, alpha_agn_grid_work)  # Normalize
    
    # Step 2: Create interpolation function for likelihood on alpha_agn
    likelihood_interp = interp1d(alpha_agn_grid_work, likelihood_alpha_agn, kind='linear', 
                                bounds_error=False, fill_value=0.0)
    
    # Debug: print range of alpha_agn_grid and likelihood
    print(f"alpha_agn_grid range: [{alpha_agn_grid_work.min():.4f}, {alpha_agn_grid_work.max():.4f}]")
    print(f"likelihood_alpha_agn range: [{likelihood_alpha_agn.min():.6f}, {likelihood_alpha_agn.max():.6f}]")
    print(f"likelihood_alpha_agn sum: {np.trapz(likelihood_alpha_agn, alpha_agn_grid_work):.6f}")
    
    # Step 3: Create grid of (f_agn, lambda_agn) values
    f_agn_grid = np.linspace(0, 1, n_fagn)
    lambda_agn_grid = np.linspace(0, 1, n_lambda)
    
    # Step 4: For each (f_agn, lambda_agn) pair, compute corresponding alpha_agn value
    # and get likelihood from interpolation
    likelihood_fagn_lambda = np.zeros((n_fagn, n_lambda))
    
    # Track alpha_agn values to check range
    alpha_agn_computed = []
    
    for i, f_agn_val in enumerate(f_agn_grid):
        for j, lambda_agn_val in enumerate(lambda_agn_grid):
            # Compute the alpha_agn for this (f_agn, lambda_agn) pair
            _, alpha_agn_val = utils.compute_gw_host_fractions(N_gal, N_agn, f_agn_val, lambda_agn=lambda_agn_val)
            alpha_agn_computed.append(alpha_agn_val)
            
            # Get likelihood from interpolation
            likelihood_fagn_lambda[i, j] = likelihood_interp(alpha_agn_val)
    
    # Debug: print range of computed alpha_agn values
    alpha_agn_computed = np.array(alpha_agn_computed)
    print(f"Computed alpha_agn range: [{alpha_agn_computed.min():.4f}, {alpha_agn_computed.max():.4f}]")
    print(f"Likelihood_fagn_lambda range before normalization: [{likelihood_fagn_lambda.min():.6f}, {likelihood_fagn_lambda.max():.6f}]")
    print(f"Number of non-zero values: {np.count_nonzero(likelihood_fagn_lambda)} / {likelihood_fagn_lambda.size}")
    
    # Debug: print range of computed alpha_agn values
    alpha_agn_computed = np.array(alpha_agn_computed)
    print(f"Computed alpha_agn range: [{alpha_agn_computed.min():.4f}, {alpha_agn_computed.max():.4f}]")
    print(f"Likelihood_fagn_lambda range before normalization: [{likelihood_fagn_lambda.min():.6f}, {likelihood_fagn_lambda.max():.6f}]")
    print(f"Number of non-zero values: {np.count_nonzero(likelihood_fagn_lambda)} / {likelihood_fagn_lambda.size}")
    
    # Check truth values if available
    if truth_f_agn is not None and truth_lambda_agn is not None:
        _, truth_alpha_agn = utils.compute_gw_host_fractions(N_gal, N_agn, truth_f_agn, lambda_agn=truth_lambda_agn)
        print(f"\nTruth values:")
        print(f"  f_agn = {truth_f_agn:.4f}, lambda_agn = {truth_lambda_agn:.4f}")
        print(f"  -> alpha_agn = {truth_alpha_agn:.4f}")
        # Find the likelihood at truth values
        truth_i = np.argmin(np.abs(f_agn_grid - truth_f_agn))
        truth_j = np.argmin(np.abs(lambda_agn_grid - truth_lambda_agn))
        truth_likelihood = likelihood_fagn_lambda[truth_i, truth_j]
        max_likelihood = np.max(likelihood_fagn_lambda)
        max_i, max_j = np.unravel_index(np.argmax(likelihood_fagn_lambda), likelihood_fagn_lambda.shape)
        print(f"  Likelihood at truth: {truth_likelihood:.6f}")
        print(f"  Max likelihood: {max_likelihood:.6f} at f_agn={f_agn_grid[max_i]:.4f}, lambda_agn={lambda_agn_grid[max_j]:.4f}")
        _, max_alpha_agn = utils.compute_gw_host_fractions(N_gal, N_agn, f_agn_grid[max_i], lambda_agn=lambda_agn_grid[max_j])
        print(f"  -> alpha_agn at max: {max_alpha_agn:.4f}")
    
    # Check if computed values are within grid range
    n_outside = np.sum((alpha_agn_computed < alpha_agn_grid_work.min()) | (alpha_agn_computed > alpha_agn_grid_work.max()))
    if n_outside > 0:
        print(f"WARNING: {n_outside} computed alpha_agn values are outside the grid range!")
        print(f"  Grid range: [{alpha_agn_grid_work.min():.4f}, {alpha_agn_grid_work.max():.4f}]")
        print(f"  Computed range: [{alpha_agn_computed.min():.4f}, {alpha_agn_computed.max():.4f}]")
    
    # Check if likelihood is all zeros (would indicate interpolation issue)
    if np.all(likelihood_fagn_lambda == 0):
        raise ValueError("All likelihood values are zero! Check if computed alpha_agn values are within grid range.")
    
    # Normalize the 2D likelihood
    likelihood_fagn_lambda = likelihood_fagn_lambda / np.trapz(np.trapz(likelihood_fagn_lambda, 
                                                                         lambda_agn_grid, axis=1), 
                                                               f_agn_grid)
    
    # Step 5: Marginalize to get 1D distributions
    # P(f_agn) = integral P(f_agn, lambda_agn) d(lambda_agn)
    likelihood_fagn = np.trapz(likelihood_fagn_lambda, lambda_agn_grid, axis=1)
    likelihood_fagn = likelihood_fagn / np.trapz(likelihood_fagn, f_agn_grid)  # Normalize
    
    # P(lambda_agn) = integral P(f_agn, lambda_agn) d(f_agn)
    likelihood_lambda = np.trapz(likelihood_fagn_lambda, f_agn_grid, axis=0)
    likelihood_lambda = likelihood_lambda / np.trapz(likelihood_lambda, lambda_agn_grid)  # Normalize
    
    # Compute summary statistics
    # For f_agn
    fagn_mean = np.trapz(f_agn_grid * likelihood_fagn, f_agn_grid)
    fagn_std = np.sqrt(np.trapz((f_agn_grid - fagn_mean)**2 * likelihood_fagn, f_agn_grid))
    # Find median
    fagn_cdf = np.cumsum(likelihood_fagn) * (f_agn_grid[1] - f_agn_grid[0])
    fagn_median = np.interp(0.5, fagn_cdf, f_agn_grid)
    
    # For lambda_agn
    lambda_mean = np.trapz(lambda_agn_grid * likelihood_lambda, lambda_agn_grid)
    lambda_std = np.sqrt(np.trapz((lambda_agn_grid - lambda_mean)**2 * likelihood_lambda, lambda_agn_grid))
    # Find median
    lambda_cdf = np.cumsum(likelihood_lambda) * (lambda_agn_grid[1] - lambda_agn_grid[0])
    lambda_median = np.interp(0.5, lambda_cdf, lambda_agn_grid)
    
    print(f"\nMarginalized distributions:")
    print(f"f_agn: mean = {fagn_mean:.4f}, std = {fagn_std:.4f}, median = {fagn_median:.4f}")
    print(f"lambda_agn: mean = {lambda_mean:.4f}, std = {lambda_std:.4f}, median = {lambda_median:.4f}")
    
    # Plot results - split into 2 separate figures for better visibility
    
    # Figure 1: 2D likelihood
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize_2d)
    im = ax1.contourf(f_agn_grid, lambda_agn_grid, likelihood_fagn_lambda.T, levels=levels, cmap=cmap)
    # Add truth values as green lines
    if truth_f_agn is not None:
        ax1.axvline(truth_f_agn, color='green', linestyle='-', linewidth=2, label=f'Truth f_agn: {truth_f_agn:.3f}')
    if truth_lambda_agn is not None:
        ax1.axhline(truth_lambda_agn, color='green', linestyle='-', linewidth=2, label=f'Truth lambda_agn: {truth_lambda_agn:.3f}')
    ax1.set_xlabel('f_agn', fontsize=12)
    ax1.set_ylabel('lambda_agn', fontsize=12)
    ax1.set_title('2D Likelihood P(f_agn, lambda_agn)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    plt.colorbar(im, ax=ax1, label='Probability Density')
    plt.tight_layout()
    plt.show()
    
    # Figure 2: 1D marginalized distributions
    fig2, axes = plt.subplots(1, 2, figsize=figsize_1d)
    
    # Plot marginalized f_agn distribution
    axes[0].plot(f_agn_grid, likelihood_fagn, 'k-', linewidth=2, label='P(f_agn)')
    # Add truth value
    if truth_f_agn is not None:
        axes[0].axvline(truth_f_agn, color='green', linestyle='-', linewidth=2, label=f'Truth: {truth_f_agn:.3f}')
    axes[0].set_xlabel('f_agn', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    axes[0].set_title('Marginalized Distribution: f_agn', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Plot marginalized lambda_agn distribution
    axes[1].plot(lambda_agn_grid, likelihood_lambda, 'k-', linewidth=2, label='P(lambda_agn)')
    # Add truth value
    if truth_lambda_agn is not None:
        axes[1].axvline(truth_lambda_agn, color='green', linestyle='-', linewidth=2, label=f'Truth: {truth_lambda_agn:.3f}')
    axes[1].set_xlabel('lambda_agn', fontsize=12)
    axes[1].set_ylabel('Probability Density', fontsize=12)
    axes[1].set_title('Marginalized Distribution: lambda_agn', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MCMC plotting functions
# ============================================================================

def get_mcmc_data(fn_config_inference):
    """
    Load MCMC data from a config inference file.
    
    Parameters:
    -----------
    fn_config_inference : str
        Path to the inference config YAML file
        
    Returns:
    --------
    dict : Dictionary containing:
        - posterior_samples: array of posterior samples
        - mcmc_params: MCMC parameter dictionary
        - truth_H0: True H0 value from config
        - truth_alpha_agn: True alpha_agn value computed from config
        - config_inference: The loaded inference config dict
        - config_data: The loaded data config dict
        - loaded_results: The full loaded results from load_inference_results()
    """
    #print(fn_config_inference)
    
    with open(fn_config_inference, 'r') as f:
        config_inference = yaml.safe_load(f)
    
    config_data_path = config_inference['fn_config_data']
    #print(config_data_path)
    
    with open(config_data_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Get output file from config
    fn_inf = config_inference['paths']['fn_inf']
    
    #print(f"Loading MCMC results from: {fn_inf}")
    
    # Load inference results
    loaded_results = run_inference.load_inference_results(fn_inf)
    
    # Extract data
    samples = loaded_results['posterior_samples']
    mcmc_params = loaded_results.get('mcmc_params', {})
    
    # Determine which parameters were varied.
    # Prefer the explicit list from the inference config if available,
    # since this is authoritative even for older result files.
    params_cfg = config_inference.get('parameters', {})
    parameters_vary_cfg = params_cfg.get('parameters_vary')
    
    if parameters_vary_cfg is not None:
        parameters_vary = list(parameters_vary_cfg)
        ndims = samples.shape[1]
        # If there is a mismatch between config and samples dimensionality,
        # fall back to inferring names from the samples themselves.
        if len(parameters_vary) != ndims:
            print(
                f"Warning: parameters_vary from config has length {len(parameters_vary)} "
                f"but posterior samples have {ndims} dimensions. Falling back to "
                "inferring parameter names from results file."
            )
            parameters_vary = None
    else:
        parameters_vary = None
    
    # If we could not get a consistent list from the config, fall back to
    # any information stored in the results file, and then to defaults.
    if parameters_vary is None:
        parameters_vary = mcmc_params.get('parameters_vary')
        if parameters_vary is None:
            # Fall back based on dimensionality if not present
            ndims = samples.shape[1]
            if ndims == 2:
                parameters_vary = ['H0', 'alpha_agn']
            elif ndims == 1:
                parameters_vary = ['alpha_agn']
            else:
                parameters_vary = [f'param_{i}' for i in range(ndims)]
    
    labels = list(parameters_vary)
    
    #print(f"Loaded MCMC results with shape: {samples.shape}")
    
    # Get truth values from config
    truth_H0 = config_data['cosmology']['H0']
    truth_f_agn = config_data['gw_injection']['f_agn']
    truth_lambda_agn = config_data['gw_injection']['lambda_agn']
    
    with h5py.File(f'{config_data["paths"]["dir_mock"]}/{config_data["paths"]["name_cat"]}', 'r') as f:
        N_gal_truth = f.attrs['n_gal']
        N_agn_truth = f.attrs['n_agn']
    
    _, truth_alpha_agn = utils.compute_gw_host_fractions(
        N_gal_truth, N_agn_truth, truth_f_agn, lambda_agn=truth_lambda_agn
    )
    
    # Build truth values ordered to match labels/parameters_vary
    truth_map = {
        'H0': truth_H0,
        'alpha_agn': truth_alpha_agn,
    }
    truth_values = [truth_map.get(name, None) for name in labels]
    
    return {
        'posterior_samples': samples,
        'mcmc_params': mcmc_params,
        'labels': labels,
        'parameters_vary': parameters_vary,
        'truth_values': truth_values,
        'truth_H0': truth_H0,
        'truth_alpha_agn': truth_alpha_agn,
        'truth_f_agn': truth_f_agn,
        'truth_lambda_agn': truth_lambda_agn,
        'config_inference': config_inference,
        'config_data': config_data,
        'loaded_results': loaded_results,
        'N_gal_truth': N_gal_truth,
        'N_agn_truth': N_agn_truth,
    }


def _mcmc_truth_values(mcmc_data, param_labels):
    """Truth values ordered to match param_labels (axis names)."""
    truth_values = mcmc_data.get('truth_values')
    if truth_values is not None:
        return truth_values
    truth_H0 = mcmc_data['truth_H0']
    truth_alpha_agn = mcmc_data['truth_alpha_agn']
    truth_map = {'H0': truth_H0, 'alpha_agn': truth_alpha_agn}
    return [truth_map.get(name, None) for name in param_labels]


def _normalize_mcmc_datasets(mcmc_data):
    """Return a list of MCMC dicts; accept a single dict or a list of dicts."""
    if isinstance(mcmc_data, dict):
        return [mcmc_data]
    return list(mcmc_data)


def _default_mcmc_overlay_colors(n):
    """Qualitative colors for n overlaid chains; tab10 for few, turbo-spaced for many."""
    if n <= 0:
        return []
    if n <= 10:
        cmap = plt.get_cmap('tab10')
        return [cmap(i % 10) for i in range(n)]
    cmap = plt.get_cmap('turbo')
    return [cmap(0.12 + 0.76 * i / max(n - 1, 1)) for i in range(n)]


def plot_mcmc_contours(mcmc_data, figsize=(4, 4), bins=50,
                       range_min=None, range_max=None, colors=None, labels=None,
                       title=None):
    """
    Plot MCMC data as a corner plot showing contours and correlations.

    Pass a single dict from get_mcmc_data(), or a list of such dicts to overplot
    all chains on one figure.

    Parameters:
    -----------
    mcmc_data : dict or list of dict
        One or more dictionaries returned by get_mcmc_data()
    figsize : tuple, optional
        Figure size for corner plot (default: (4, 4))
    bins : int, optional
        Number of bins for histograms (default: 50)
    range_min, range_max : float, optional
        If set, applied to every parameter axis (same as before).
    colors : list, optional
        Matplotlib colors, one per dataset when ``mcmc_data`` is a list (or one
        color for a single dict). If omitted, a tab10 / turbo colormap is used.
    labels : list of str, optional
        Legend label for each overlaid dataset (same length as the number of
        runs). If omitted, chain contours/histograms are not listed in the
        legend (truths are still shown when present).
    title : str, optional
        Figure title (suptitle for corner plots; axes title for the 1D fallback).
    """
    datasets = _normalize_mcmc_datasets(mcmc_data)
    if not datasets:
        return

    param_labels = datasets[0]['labels']
    n_params = datasets[0]['posterior_samples'].shape[1]

    for k, ds in enumerate(datasets):
        if ds['labels'] != param_labels:
            raise ValueError(
                f"mcmc_data[{k}] has labels {ds['labels']!r}; expected {param_labels!r}"
            )
        if ds['posterior_samples'].shape[1] != n_params:
            raise ValueError(
                f"mcmc_data[{k}] has {ds['posterior_samples'].shape[1]} parameters; "
                f"expected {n_params}"
            )

    n_runs = len(datasets)
    if colors is not None and len(colors) != n_runs:
        raise ValueError(f"colors length {len(colors)} != number of datasets {n_runs}")
    if labels is not None and len(labels) != n_runs:
        raise ValueError(f"labels length {len(labels)} != number of datasets {n_runs}")

    if colors is None:
        colors = (
            ['black'] if n_runs == 1 else _default_mcmc_overlay_colors(n_runs)
        )
    legend_labels = labels

    truth_values = _mcmc_truth_values(datasets[0], param_labels)

    # For 1D posteriors, use overlaid marginalized histograms
    if n_params == 1:
        print("Only 1 parameter varied; using 1D marginalized plot instead of corner.")
        fig, ax = plt.subplots(figsize=figsize)
        label_1d = param_labels[0]
        tv = truth_values[0]

        if range_min is not None and range_max is not None:
            bin_edges = np.linspace(range_min, range_max, bins)
        else:
            bin_edges = bins

        hist_alpha = 0.7 if n_runs > 1 else 1.0
        for i, ds in enumerate(datasets):
            s = ds['posterior_samples'][:, 0]
            ax.hist(
                s,
                bins=bin_edges,
                density=True,
                histtype='step',
                lw=2,
                color=colors[i],
                alpha=hist_alpha,
                label=legend_labels[i] if legend_labels is not None else None,
            )
        if tv is not None:
            ax.axvline(
                tv,
                color='green',
                linestyle='-',
                linewidth=2,
                label=f'Truth: {tv:.3f}',
            )
        ax.set_xlabel(label_1d)
        ax.set_ylabel('Density')
        ax.set_title(title if title is not None else f'Posterior: {label_1d}')
        if legend_labels is not None or tv is not None:
            ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    try:
        import corner
        from matplotlib.patches import Patch
    except ImportError:
        print("corner package not available, skipping corner plot")
        return

    ranges = []
    buffer_factor = 0.01

    for i in range(n_params):
        sample_mins = []
        sample_maxs = []
        for ds in datasets:
            s = ds['posterior_samples']
            sample_mins.append(np.min(s[:, i]))
            sample_maxs.append(np.max(s[:, i]))
        lo = min(sample_mins)
        hi = max(sample_maxs)
        tv = truth_values[i]
        if tv is not None:
            lo = min(lo, tv)
            hi = max(hi, tv)
        if range_min is not None:
            lo = range_min
        if range_max is not None:
            hi = range_max
        span = hi - lo
        if span <= 0:
            span = max(abs(hi), 1.0) * 0.05 + 1e-12
        ranges.append([lo - buffer_factor * span, hi + buffer_factor * span])

    fig = None
    for i, ds in enumerate(datasets):
        samples = ds['posterior_samples']
        if fig is None:
            fig = corner.corner(
                samples,
                labels=param_labels,
                show_titles=True,
                title_kwargs={"fontsize": 12},
                bins=bins,
                range=ranges,
                figsize=figsize,
                color=colors[i],
                truths=truth_values,
                truth_color='green',
                truth_kwargs={'linestyle': '-', 'linewidth': 2},
            )
        else:
            fig = corner.corner(
                samples,
                fig=fig,
                bins=bins,
                range=ranges,
                color=colors[i],
                show_titles=False,
            )

    if legend_labels is not None:
        handles = [
            Patch(facecolor=colors[i], edgecolor=colors[i], label=legend_labels[i])
            for i in range(n_runs)
        ]
        fig.legend(
            handles=handles,
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            framealpha=0.9,
        )

    if title is not None:
        fig.suptitle(title, fontsize=13)
        fig.subplots_adjust(top=0.92)

    plt.show()


def plot_mcmc_marginalized(mcmc_data, figsize=(12, 5), bins=50,
                           range_min=None, range_max=None):
    """
    Plot marginalized posterior distributions for each parameter.
    
    Parameters:
    -----------
    mcmc_data : dict
        Dictionary returned by get_mcmc_data()
    figsize : tuple, optional
        Figure size for histograms (default: (12, 5))
    bins : int, optional
        Number of bins for histograms (default: 50)
    """
    samples = mcmc_data['posterior_samples']
    labels = mcmc_data['labels']
    # Truth values ordered to match labels/parameters_vary
    truth_values = mcmc_data.get('truth_values')
    if truth_values is None:
        truth_H0 = mcmc_data['truth_H0']
        truth_alpha_agn = mcmc_data['truth_alpha_agn']
        truth_map = {'H0': truth_H0, 'alpha_agn': truth_alpha_agn}
        truth_values = [truth_map.get(name, None) for name in labels]
    
    # Plot histograms for each parameter
    n_params = samples.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=figsize)
    
    if n_params == 1:
        axes = [axes]
    
    if range_min is not None and range_max is not None:
        bins = np.linspace(range_min, range_max, bins)
    else:
        bins = bins
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.hist(samples[:, i], bins=bins, density=True, lw=2, color='black', histtype='step')
        # Overplot truth value
        ax.axvline(truth_values[i], color='green', linestyle='-', linewidth=2,
                   label=f'Truth: {truth_values[i]:.3f}')
        # if range_min is not None and range_max is not None:
        #     ax.set_xlim(range_min, range_max)
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior: {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_mcmc_data(mcmc_data, figsize_hist=(12, 5), figsize_corner=(4, 4), bins=50):
    """
    Plot MCMC data as contours and marginalized distributions.
    
    This function calls plot_mcmc_contours() first, then plot_mcmc_marginalized().
    
    Parameters:
    -----------
    mcmc_data : dict
        Dictionary returned by get_mcmc_data()
    figsize_hist : tuple, optional
        Figure size for histograms (default: (12, 5))
    figsize_corner : tuple, optional
        Figure size for corner plot (default: (10, 10))
    bins : int, optional
        Number of bins for histograms (default: 50)
    """
    # Plot contours first
    plot_mcmc_contours(mcmc_data, figsize=figsize_corner, bins=bins)
    
    # Then plot marginalized distributions
    plot_mcmc_marginalized(mcmc_data, figsize=figsize_hist, bins=bins)


def plot_mcmc_fagn_lambdaagn(mcmc_data, figsize=(12, 5), bins=50):
    """
    Plot marginalized posterior distributions for f_agn and lambda_agn.
    
    Parameters:
    -----------
    mcmc_data : dict
        Dictionary returned by get_mcmc_data()
    figsize : tuple, optional
    """
    ### TODO this isn't the right way to do this

    samples = mcmc_data['posterior_samples']
    labels = mcmc_data['labels']
    truth_H0 = mcmc_data['truth_H0']
    truth_alpha_agn = mcmc_data['truth_alpha_agn']
    truth_f_agn = mcmc_data['truth_f_agn']
    truth_lambda_agn = mcmc_data['truth_lambda_agn']

    # Solve for f_agn and lambda_agn from posterior f samples and plot
    frac_agn_samples = samples[:, 1]  # f corresponds to frac_agn in this inference
    N_gal = mcmc_data['N_gal_truth']
    N_agn = mcmc_data['N_agn_truth']
    fagn_solutions = []
    lambda_solutions = []
    for ff in frac_agn_samples:
        f_sol, lam_sol = run_inference.solve_fagn_lambda(ff, N_gal, N_agn)
        fagn_solutions.append(f_sol)
        lambda_solutions.append(lam_sol)

    fagn_solutions = np.array(fagn_solutions)
    lambda_solutions = np.array(lambda_solutions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(fagn_solutions, bins=50, color='k', histtype='step', lw=2)
    axes[0].axvline(truth_f_agn, color='green', linestyle='-', linewidth=2, label=f"Truth: {truth_f_agn:.3f}")
    axes[0].set_xlabel('f_agn')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(lambda_solutions, bins=50, color='k', histtype='step', lw=2)
    axes[1].axvline(truth_lambda_agn, color='green', linestyle='-', linewidth=2, label=f"Truth: {truth_lambda_agn:.3f}")
    axes[1].set_xlabel('lambda_agn')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_mcmc_fagn_lambdaagn_grid(mcmc_data, n_fagn=100, n_lambda=100, 
                                    figsize_2d=(7, 5), figsize_1d=(16, 6),
                                    levels=20, cmap='viridis', sigma_tol=0.01):
    """
    Plot posterior distributions for f_agn and lambda_agn from MCMC samples of alpha_agn.
    
    This function handles the degeneracy between f_agn and lambda_agn by:
    1. Creating a grid of (f_agn, lambda_agn) values
    2. For each alpha_agn sample, computing which grid points are consistent
    3. Building up a posterior density over the grid
    4. Plotting 2D contours and marginalized 1D distributions
    
    Parameters:
    -----------
    mcmc_data : dict
        Dictionary returned by get_mcmc_data()
    n_fagn : int, optional
        Number of grid points for f_agn (default: 100)
    n_lambda : int, optional
        Number of grid points for lambda_agn (default: 100)
    figsize_2d : tuple, optional
        Figure size for 2D plot (default: (10, 8))
    figsize_1d : tuple, optional
        Figure size for 1D plots (default: (16, 6))
    levels : int, optional
        Number of contour levels (default: 20)
    cmap : str, optional
        Colormap name (default: 'viridis')
    sigma_tol : float, optional
        Tolerance for matching alpha_agn values, as fraction of typical spread (default: 0.01)
    """
    samples = mcmc_data['posterior_samples']
    labels = mcmc_data['labels']
    truth_f_agn = mcmc_data['truth_f_agn']
    truth_lambda_agn = mcmc_data['truth_lambda_agn']
    N_gal = mcmc_data['N_gal_truth']
    N_agn = mcmc_data['N_agn_truth']
    
    # Extract alpha_agn samples using label-based indexing
    if 'alpha_agn' not in labels:
        raise ValueError(
            "plot_mcmc_fagn_lambdaagn_grid requires 'alpha_agn' to be one of the "
            f"MCMC parameters, but labels are: {labels}. "
            "Run an inference where alpha_agn is varied to use this function."
        )
    alpha_idx = labels.index('alpha_agn')
    alpha_agn_samples = samples[:, alpha_idx]
    
    # Create grid of (f_agn, lambda_agn) values
    f_agn_grid = np.linspace(0, 1, n_fagn)
    lambda_agn_grid = np.linspace(0, 1, n_lambda)
    F_agn, Lambda_agn = np.meshgrid(f_agn_grid, lambda_agn_grid)
    
    # Compute alpha_agn for each grid point (vectorized)
    # Flatten grids for vectorized computation
    F_agn_flat = F_agn.flatten()
    Lambda_agn_flat = Lambda_agn.flatten()
    alpha_agn_flat = np.zeros(len(F_agn_flat))
    
    for idx in range(len(F_agn_flat)):
        _, alpha_agn_flat[idx] = utils.compute_gw_host_fractions(
            N_gal, N_agn, F_agn_flat[idx], Lambda_agn_flat[idx]
        )
    
    # Reshape back to grid
    alpha_agn_grid = alpha_agn_flat.reshape(F_agn.shape)
    
    # Build posterior density
    # For each alpha_agn sample, find grid points that are consistent
    # Use a Gaussian kernel to weight contributions
    posterior_density = np.zeros_like(alpha_agn_grid)
    
    # Estimate typical spread of alpha_agn samples
    alpha_std = np.std(alpha_agn_samples)
    tol = sigma_tol * alpha_std
    
    # For each sample, add contribution to nearby grid points
    for alpha_sample in alpha_agn_samples:
        # Find grid points where computed alpha_agn is close to sample
        diff = np.abs(alpha_agn_grid - alpha_sample)
        # Use Gaussian weighting
        weights = np.exp(-0.5 * (diff / tol)**2)
        posterior_density += weights
    
    # Normalize
    posterior_density /= np.sum(posterior_density) * (f_agn_grid[1] - f_agn_grid[0]) * (lambda_agn_grid[1] - lambda_agn_grid[0])
    
    # Marginalize to get 1D distributions
    likelihood_fagn = np.trapz(posterior_density, lambda_agn_grid, axis=0)
    likelihood_lambda = np.trapz(posterior_density, f_agn_grid, axis=1)
    
    # Normalize 1D distributions
    likelihood_fagn /= np.trapz(likelihood_fagn, f_agn_grid)
    likelihood_lambda /= np.trapz(likelihood_lambda, lambda_agn_grid)
    
    # Compute statistics
    fagn_mean = np.trapz(f_agn_grid * likelihood_fagn, f_agn_grid)
    fagn_std = np.sqrt(np.trapz((f_agn_grid - fagn_mean)**2 * likelihood_fagn, f_agn_grid))
    fagn_cdf = np.cumsum(likelihood_fagn) * (f_agn_grid[1] - f_agn_grid[0])
    fagn_median = np.interp(0.5, fagn_cdf, f_agn_grid)
    
    lambda_mean = np.trapz(lambda_agn_grid * likelihood_lambda, lambda_agn_grid)
    lambda_std = np.sqrt(np.trapz((lambda_agn_grid - lambda_mean)**2 * likelihood_lambda, lambda_agn_grid))
    lambda_cdf = np.cumsum(likelihood_lambda) * (lambda_agn_grid[1] - lambda_agn_grid[0])
    lambda_median = np.interp(0.5, lambda_cdf, lambda_agn_grid)
    
    print(f"\nMarginalized distributions:")
    print(f"f_agn: mean = {fagn_mean:.4f}, std = {fagn_std:.4f}, median = {fagn_median:.4f}")
    print(f"lambda_agn: mean = {lambda_mean:.4f}, std = {lambda_std:.4f}, median = {lambda_median:.4f}")
    
    # Plot results - split into 2 separate figures
    
    # Figure 1: 2D posterior
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize_2d)
    # posterior_density has shape (n_lambda, n_fagn); use transpose here to
    # preserve the original visual orientation used in earlier analyses.
    # TODO triple check the transpose here...
    im = ax1.contourf(f_agn_grid, lambda_agn_grid, posterior_density, levels=levels, cmap=cmap)
    # Add truth values
    if truth_f_agn is not None:
        ax1.axvline(truth_f_agn, color='green', linestyle='-', linewidth=2, label=f'Truth f_agn: {truth_f_agn:.3f}')
    if truth_lambda_agn is not None:
        ax1.axhline(truth_lambda_agn, color='green', linestyle='-', linewidth=2, label=f'Truth lambda_agn: {truth_lambda_agn:.3f}')
    if truth_f_agn is not None and truth_lambda_agn is not None:
        ax1.scatter(truth_f_agn, truth_lambda_agn, color='green', marker='s', s=100, zorder=10)
    ax1.set_xlabel('f_agn', fontsize=12)
    ax1.set_ylabel('lambda_agn', fontsize=12)
    ax1.set_title('2D Posterior P(f_agn, lambda_agn | alpha_agn samples)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    plt.colorbar(im, ax=ax1, label='Probability Density')
    plt.tight_layout()
    plt.show()
    
    # Figure 2: 1D marginalized distributions
    fig2, axes = plt.subplots(1, 2, figsize=figsize_1d)
    
    # Plot marginalized f_agn distribution
    axes[0].plot(f_agn_grid, likelihood_fagn, 'k-', linewidth=2, label='P(f_agn)')
    if truth_f_agn is not None:
        axes[0].axvline(truth_f_agn, color='green', linestyle='-', linewidth=2, label=f'Truth: {truth_f_agn:.3f}')
    axes[0].set_xlabel('f_agn', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    axes[0].set_title('Marginalized Distribution: f_agn', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Plot marginalized lambda_agn distribution
    axes[1].plot(lambda_agn_grid, likelihood_lambda, 'k-', linewidth=2, label='P(lambda_agn)')
    if truth_lambda_agn is not None:
        axes[1].axvline(truth_lambda_agn, color='green', linestyle='-', linewidth=2, label=f'Truth: {truth_lambda_agn:.3f}')
    axes[1].set_xlabel('lambda_agn', fontsize=12)
    axes[1].set_ylabel('Probability Density', fontsize=12)
    axes[1].set_title('Marginalized Distribution: lambda_agn', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
