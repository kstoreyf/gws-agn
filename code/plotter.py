"""
Plotting utilities for GW-AGN analysis.

This module provides functions for:
- Sky plots (RA/Dec visualization)
- Likelihood grid plotting
- MCMC posterior plotting
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import yaml

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
        'config_inference': config_inference,
        'config_data': config_data
    }


def plot_likelihood_data(likelihood_data, figsize=(7, 6)):
    """
    Plot likelihood data as a contour plot.
    
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
    
    # Create meshgrid for plotting
    H0_mesh, alpha_agn_mesh = np.meshgrid(H0_grid, alpha_agn_grid)
    
    # Plot likelihood grid as contour plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to likelihood (relative, normalized) or use log directly
    likelihood_grid = np.exp(log_likelihood_grid - np.nanmax(log_likelihood_grid))
    
    # Apply transpose for plotting
    likelihood_grid = likelihood_grid.T
    
    # Fill contours
    im = ax.contourf(H0_mesh, alpha_agn_mesh, likelihood_grid, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Relative Likelihood')
    
    # Add green truth lines
    ax.axvline(truth_H0, color='green', linestyle='-', linewidth=1, label=f'Truth H0: {truth_H0:.1f}')
    ax.axhline(truth_alpha_agn, color='green', linestyle='-', linewidth=1, label=f'Truth α_AGN: {truth_alpha_agn:.3f}')
    
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]', fontsize=14)
    ax.set_ylabel(r'$\alpha_{AGN}$', fontsize=14)
    ax.set_title(f'Likelihood Grid', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
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
    print(fn_config_inference)
    
    with open(fn_config_inference, 'r') as f:
        config_inference = yaml.safe_load(f)
    
    config_data_path = config_inference['fn_config_data']
    print(config_data_path)
    
    with open(config_data_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Get output file from config
    fn_inf = config_inference['paths']['fn_inf']
    
    print(f"Loading MCMC results from: {fn_inf}")
    
    # Load inference results
    loaded_results = run_inference.load_inference_results(fn_inf)
    
    # Extract data
    samples = loaded_results['posterior_samples']
    mcmc_params = loaded_results.get('mcmc_params', {})
    labels = mcmc_params.get('labels', ['H0', 'alpha_agn'])
    
    print(f"Loaded MCMC results with shape: {samples.shape}")
    
    # Get truth values from config
    truth_H0 = config_data['cosmology']['H0']
    truth_f_agn = config_data['gw_injection']['f_agn']
    truth_lambda_agn = config_data['gw_injection']['lambda_agn']
    
    with h5py.File(f'{config_data["paths"]["dir_mock"]}/{config_data["paths"]["name_cat"]}', 'r') as f:
        N_gal_truth = f.attrs['n_gal']
        N_agn_truth = f.attrs['n_agn']
    
    _, truth_alpha_agn = utils.compute_gw_host_fractions(N_gal_truth, N_agn_truth, truth_f_agn, lambda_agn=truth_lambda_agn)
    
    return {
        'posterior_samples': samples,
        'mcmc_params': mcmc_params,
        'labels': labels,
        'truth_H0': truth_H0,
        'truth_alpha_agn': truth_alpha_agn,
        'config_inference': config_inference,
        'config_data': config_data,
        'loaded_results': loaded_results
    }


def plot_mcmc_contours(mcmc_data, figsize=(10, 10), bins=50):
    """
    Plot MCMC data as a corner plot showing contours and correlations.
    
    Parameters:
    -----------
    mcmc_data : dict
        Dictionary returned by get_mcmc_data()
    figsize : tuple, optional
        Figure size for corner plot (default: (10, 10))
    bins : int, optional
        Number of bins for histograms (default: 50)
    """
    samples = mcmc_data['posterior_samples']
    labels = mcmc_data['labels']
    truth_H0 = mcmc_data['truth_H0']
    truth_alpha_agn = mcmc_data['truth_alpha_agn']
    
    truth_values = [truth_H0, truth_alpha_agn]
    n_params = samples.shape[1]
    
    # Corner plot
    try:
        import corner
    except ImportError:
        print("corner package not available, skipping corner plot")
        return
    
    # Calculate ranges for each parameter that include truth values and majority of samples
    ranges = []
    buffer_factor = 0.01  # 5% buffer on each side
    
    for i in range(n_params):
        sample_min = np.min(samples[:, i])
        sample_max = np.max(samples[:, i])
        # sample_min = np.percentile(samples[:, i], 16)
        # sample_max = np.percentile(samples[:, i], 84)
        print(sample_min, sample_max)
        truth_val = truth_values[i]
        
        # Ensure range includes truth value
        range_min = min(sample_min, truth_val) * (1 - buffer_factor)
        range_max = max(sample_max, truth_val) * (1 + buffer_factor)
        print(range_min, range_max)
        ranges.append([range_min, range_max])
    
    fig = corner.corner(samples, labels=labels, show_titles=True, 
                        title_kwargs={"fontsize": 12}, bins=bins,
                        truths=truth_values, truth_color='green', 
                        truth_kwargs={'linestyle': '-', 'linewidth': 2},
                        range=ranges, figsize=figsize)
    plt.show()


def plot_mcmc_marginalized(mcmc_data, figsize=(12, 5), bins=50):
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
    truth_H0 = mcmc_data['truth_H0']
    truth_alpha_agn = mcmc_data['truth_alpha_agn']
    
    truth_values = [truth_H0, truth_alpha_agn]
    
    # Plot histograms for each parameter
    n_params = samples.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=figsize)
    
    if n_params == 1:
        axes = [axes]
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.hist(samples[:, i], bins=bins, density=True, lw=2, color='black', histtype='step')
        # Overplot truth value
        ax.axvline(truth_values[i], color='green', linestyle='-', linewidth=2,
                   label=f'Truth: {truth_values[i]:.3f}')
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior: {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_mcmc_data(mcmc_data, figsize_hist=(12, 5), figsize_corner=(10, 10), bins=50):
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