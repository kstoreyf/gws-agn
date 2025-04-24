import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    
    plt.legend()
    
    return fig, ax