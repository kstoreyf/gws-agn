#!/usr/bin/env python3
"""
Preprocessing script for lognormal catalogs
Converts the catalog into a pixelated format for inference
"""

# Disable JAX memory preallocation to avoid memory issues
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import argparse
import yaml

# JAX imports for numerical computing (though most of this script uses NumPy)
import jax
from jax import random, jit, vmap, grad
from jax import numpy as jnp
from jax.lax import cond

# Standard scientific computing libraries
import astropy
import numpy as np
import healpy as hp  # For HEALPix pixelation of the sky
import h5py  # For reading/writing HDF5 files
import astropy.units as u

# Cosmology and interpolation utilities (imported but may not all be used in this script)
from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
import astropy.constants as constants
from jax.scipy.special import logsumexp
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from tqdm import tqdm  # For progress bars

# Configure JAX to use 64-bit precision and highest precision matrix multiplication
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'highest')

# Import jaxinterp2d for 2D interpolation (may not be used in this script)
from jaxinterp2d import interp2d, CartesianGrid


def parse_args():
    """
    Parse command line arguments for config file.
    
    Returns:
    --------
    config : dict
        Configuration dictionary loaded from YAML file
    """
    parser = argparse.ArgumentParser(description='Pixelize catalogs for inference')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # Load YAML config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main(config):
    """
    Main function to pixelize catalogs.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from YAML file
    """
    import time
    t_start = time.perf_counter()
    
    print("Starting preprocessing...")
    
    # Extract parameters from config
    dir_mock = config['paths']['dir_mock']
    catalog_filename = config['paths']['name_cat']
    file = os.path.join(dir_mock, catalog_filename)
    print(f"Loading catalog from {dir_mock}")
    
    # Set up HEALPix pixelation scheme
    nside = config['pixelization']['nside']
    npix = hp.pixelfunc.nside2npix(nside)  # Total number of pixels
    print(f"Number of pixels: {npix}")
    
    # Process galaxies
    # max_sources will be calculated dynamically from the data
    print("Processing galaxies...")
    ras_gal, decs_gal, zs_gal = load_catalog_data(file, 'gal')
    print(f"Loaded {len(ras_gal)} galaxies")
    
    z_gal_pixelated, n_gal_per_pixel, max_sources_gal = process_objects_to_pixels(
        ras_gal, decs_gal, zs_gal, nside, max_sources=None, desc="Processing galaxies"
    )
    
    print("Saving galaxy data...")
    fn_cat_gal_pixelated = os.path.join(dir_mock, config['paths']['name_cat_gal_pixelated'])
    save_pixelated_data(fn_cat_gal_pixelated, nside, z_gal_pixelated, n_gal_per_pixel, 'galaxies')
    
    # Process AGN (Active Galactic Nuclei) - same procedure as galaxies
    # max_sources will be calculated dynamically (may be different from galaxies)
    print("Processing AGN...")
    ras_agn, decs_agn, zs_agn = load_catalog_data(file, 'agn')
    print(f"Loaded {len(ras_agn)} AGN")
    
    z_agn_pixelated, n_agn_per_pixel, max_sources_agn = process_objects_to_pixels(
        ras_agn, decs_agn, zs_agn, nside, max_sources=None, desc="Processing AGN"
    )
    
    print("Saving AGN data...")
    fn_cat_agn_pixelated = os.path.join(dir_mock, config['paths']['name_cat_agn_pixelated'])
    save_pixelated_data(fn_cat_agn_pixelated, nside, z_agn_pixelated, n_agn_per_pixel, 'agn')
    
    print(f"\nSummary:")
    print(f"  Max galaxies per pixel: {max_sources_gal}")
    print(f"  Max AGN per pixel: {max_sources_agn}")
    print("Preprocessing finished successfully!")
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    minutes = elapsed / 60
    print(f"Total time: {elapsed:.2f} s = {minutes:.2f} min")
    

def load_catalog_data(filepath, object_type):
    """
    Load catalog data (galaxies or AGN) from HDF5 file.
    
    Parameters:
    -----------
    filepath : str
        Path to the HDF5 catalog file
    object_type : str
        Either 'gal' for galaxies or 'agn' for AGN
        
    Returns:
    --------
    ras : numpy.ndarray
        Right ascension in radians
    decs : numpy.ndarray
        Declination in radians
    zs : numpy.ndarray
        Redshifts
    """
    with h5py.File(filepath, 'r') as f:
        ras = np.asarray(f[f'ra_{object_type}'])*np.pi/180  # Convert degrees to radians
        decs = np.asarray(f[f'dec_{object_type}'])*np.pi/180  # Convert degrees to radians
        zs = np.asarray(f[f'z_{object_type}'])  # Redshifts
        
    return ras, decs, zs


def calculate_max_sources_per_pixel(ras, decs, zs, nside):
    """
    Calculate the maximum number of sources in any single pixel.
    
    Parameters:
    -----------
    ras : numpy.ndarray
        Right ascension in radians
    decs : numpy.ndarray
        Declination in radians
    zs : numpy.ndarray
        Redshifts
    nside : int
        HEALPix resolution parameter
        
    Returns:
    --------
    max_sources : int
        Maximum number of sources in any pixel
    """
    # Convert sky positions (RA, Dec) to HEALPix pixel indices
    # Note: HEALPix uses colatitude (pi/2 - dec) instead of declination
    pixel_indices = hp.pixelfunc.ang2pix(nside, np.pi/2-decs, ras)
    
    # Count sources per pixel
    npix = hp.pixelfunc.nside2npix(nside)
    n_sources_per_pixel = np.bincount(pixel_indices, minlength=npix)
    
    # Return the maximum count
    return int(n_sources_per_pixel.max())


def process_objects_to_pixels(ras, decs, zs, nside, max_sources=None, desc="Processing objects"):
    """
    Process sources (galaxies or AGN) into HEALPix pixels with padding.
    
    Parameters:
    -----------
    ras : numpy.ndarray
        Right ascension in radians
    decs : numpy.ndarray
        Declination in radians
    zs : numpy.ndarray
        Redshifts
    nside : int
        HEALPix resolution parameter
    max_sources : int, optional
        Maximum number of sources per pixel (for padding). If None, calculated dynamically.
    desc : str
        Description for progress bar
        
    Returns:
    --------
    z_per_pixel : list
        List of arrays, one per pixel, containing redshifts (padded to max_sources)
    n_per_pixel : list
        List of actual source counts per pixel
    max_sources : int
        The max_sources value used (either provided or calculated)
    """
    # Calculate max_sources dynamically if not provided
    if max_sources is None:
        print(f"Calculating maximum sources per pixel for {desc}...")
        max_sources = calculate_max_sources_per_pixel(ras, decs, zs, nside)
        print(f"Maximum sources per pixel: {max_sources}")
    
    # Set up HEALPix pixelation scheme
    npix = hp.pixelfunc.nside2npix(nside)  # Total number of pixels
    pixel_grid = np.arange(npix)  # Array of all pixel indices
    
    # Convert sky positions (RA, Dec) to HEALPix pixel indices
    # Note: HEALPix uses colatitude (pi/2 - dec) instead of declination
    pixel_indices = hp.pixelfunc.ang2pix(nside, np.pi/2-decs, ras)
    
    # Process sources into pixels
    z_per_pixel = []  # List to store redshift arrays for each pixel
    n_per_pixel = []  # List to store actual number of sources per pixel
    
    # Loop through each pixel and group sources by their pixel assignment
    for pix in tqdm(pixel_grid, desc=desc):
        # Find all sources that fall in this pixel
        idx_in_pixel = np.where(pixel_indices == pix)[0]
        z_in_pixel = zs[idx_in_pixel]  # Get redshifts of sources in this pixel
        n_sources_in_pixel = z_in_pixel.shape[0]  # Count how many sources are in this pixel
        z_with_padding = [z_in_pixel]  # Start with the actual source redshifts
        
        # Pad with NaN values if pixel has fewer than max_sources
        # This ensures all pixels have the same array size for efficient batch processing
        if n_sources_in_pixel < max_sources:
            n_padding = int(max_sources - n_sources_in_pixel)
            z_with_padding.append(np.full(n_padding, np.nan))  # Use NaN as a sentinel value for missing data
        
        # Concatenate real redshifts with padding, store in z_per_pixel
        z_per_pixel.append(np.concatenate(z_with_padding))
        n_per_pixel.append(n_sources_in_pixel)  # Store the actual count for later reference
    
    return z_per_pixel, n_per_pixel, max_sources


def save_pixelated_data(fn_cat_pixelated, nside, z_per_pixel, n_per_pixel, source_type):
    """
    Save pixelated data to HDF5 file.
    
    Parameters:
    -----------
    fn_cat_pixelated : str
        Full path to output file
    nside : int
        HEALPix resolution parameter
    z_per_pixel : list
        List of arrays containing redshifts per pixel (padded)
    n_per_pixel : list
        List of actual source counts per pixel
    source_type : str
        Source type string (e.g., 'galaxies' or 'agn') used for logging
    """
    n_per_pixel_array = np.asarray(n_per_pixel)  # Convert to numpy array
    
    print(f"Max {source_type} per pixel: {n_per_pixel_array.max()}")
    
    # Write pixelated data to HDF5 file
    # Use simplified dataset names: 'z' and 'n_in_pixel'
    with h5py.File(fn_cat_pixelated, 'w') as f:
        f.attrs['nside'] = nside  # Store HEALPix resolution as metadata
        f.create_dataset('z', data=np.asarray(z_per_pixel), compression='gzip', shuffle=False)
        f.create_dataset('n_in_pixel', data=n_per_pixel_array, compression='gzip', shuffle=False)
    
    print(f"{source_type.capitalize()} preprocessing complete!")


if __name__ == "__main__":
    config = parse_args()
    main(config) 