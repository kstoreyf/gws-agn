"""
Script to generate gravitational wave (GW) samples from mock galaxy and AGN catalogs.
This script:
1. Loads mock catalog data (galaxies and AGNs with positions and redshifts)
2. Sets up cosmological distance calculations
3. Generates GW event samples with position and luminosity distance uncertainties
4. Saves the samples to an HDF5 file for later inference
"""

import os
import sys

# Set environment variables BEFORE any imports
# Disable XLA preallocation to reduce memory usage
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# Force JAX to use CPU (ksf adding this)
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Fix for coordination_agent_recoverable flag conflict
# Add --undefok flag to sys.argv BEFORE any absl imports
# This prevents errors when TensorFlow/JAX libraries try to redefine flags
# if '--undefok' not in ' '.join(sys.argv):
#     sys.argv.insert(1, '--undefok=coordination_agent_recoverable')

# Parse absl flags early with undefok to prevent redefinition errors
try:
    from absl import flags
    # Parse with undefok before any TensorFlow/JAX initialization
    flags.FLAGS(sys.argv, known_only=True)
except (Exception, SystemExit):
    # Ignore flag parsing errors at this stage
    pass

import warnings
warnings.filterwarnings('ignore')

import argparse
import yaml

import jax

# JAX imports for numerical computation and automatic differentiation
from jax import random, jit, vmap, grad
from jax import numpy as jnp
from jax.lax import cond

# Astronomy and cosmology libraries
import astropy
import numpy as np
import healpy as hp  # For HEALPix pixelization (not used in this script but imported)

# File I/O and units
import h5py
import astropy.units as u

# Cosmology and physical constants
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as constants
from jax.scipy.special import logsumexp
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from tqdm import tqdm  # Progress bars
from jaxinterp2d import interp2d, CartesianGrid  # 2D interpolation for cosmology
from scipy.stats import multivariate_normal  # For generating correlated samples

# Configure JAX for high precision calculations
jax.config.update("jax_enable_x64", True)  # Use 64-bit floats for accuracy
jax.config.update('jax_default_matmul_precision', 'highest')  # Highest precision matrix multiplication


def parse_args():
    """
    Parse command line arguments for config file.
    
    Returns:
    --------
    config : dict
        Configuration dictionary loaded from YAML file
    overwrite : bool
        Whether to overwrite existing GW samples
    """
    parser = argparse.ArgumentParser(description='Generate GW samples from mock catalogs')
    parser.add_argument('config', type=str, nargs='?', help='Path to YAML configuration file')
    parser.add_argument('--config', dest='config_flag', type=str,
                        help='Path to YAML configuration file (legacy flag)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing GW samples if they exist')
    args = parser.parse_args()
    
    # Prefer positional config, fall back to --config for backwards compatibility
    config_path = args.config or args.config_flag
    if not config_path:
        parser.error('Please provide the config file as the first argument or via --config.')

    # Load YAML config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config, args.overwrite


def main(config, overwrite=False):
    """
    Main function to generate GW samples from mock catalogs.
    Always includes black hole masses in the samples.
    All samples are saved as 2D arrays with shape (N_gw, N_samples_gw).
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary from YAML file
    """
    import time
    t_start = time.perf_counter()
    
    # Extract parameters from config
    
    # File paths for input data
    dir_mock = config['paths']['dir_mock']
    fn_cat = os.path.join(dir_mock, config['paths']['name_cat'])
    fn_gw = os.path.join(dir_mock, config['paths']['name_gw'])
    fn_gwsamples = os.path.join(dir_mock, config['paths']['name_gwsamples'])

    if not overwrite and os.path.exists(fn_gwsamples):
        print("GW samples already exist. Skipping. Use --overwrite to regenerate:")
        print(f"  {fn_gwsamples}")
        return

    # Load data
    ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, i_gw_gal, i_gw_agn = load_data(fn_cat, fn_gw)
    
    # Number of samples to generate per GW event
    N_samples_gw = config['gw_samples']['N_samples_gw']

    # Redshift grid boundaries for cosmology calculations (fixed values)
    zMax_1 = 0.5  # Low redshift boundary
    zMax_2 = 5.0  # High redshift boundary

    # Cosmological parameters from config
    H0 = config['cosmology']['H0']
    Om0 = config['cosmology']['Om0']

    # Set up cosmology
    zgrid, rs, Om0grid, speed_of_light = setup_cosmology(zMax_1, zMax_2, H0, Om0)
    
    # Create cosmology functions
    cosmo_funcs = create_cosmology_functions(Om0grid, zgrid, rs, Om0, H0, speed_of_light)
    dL_of_z = cosmo_funcs['dL_of_z']

    # Extract GW event positions and compute distances
    ra_gal_gw, dec_gal_gw, dL_gal_gw, ra_agn_gw, dec_agn_gw, dL_agn_gw = extract_gw_positions(
        ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, 
        i_gw_gal, i_gw_agn, dL_of_z, H0
    )

    # Generate black hole masses
    mass_mean = config['gw_samples']['mass_mean']
    mass_std = config['gw_samples']['mass_std']
    m1sdet_gal_gw, m2sdet_gal_gw = generate_black_hole_masses(z_gal[i_gw_gal], len(i_gw_gal), 
                                                               mass_mean=mass_mean, mass_std=mass_std)
    m1sdet_agn_gw, m2sdet_agn_gw = generate_black_hole_masses(z_agn[i_gw_agn], len(i_gw_agn),
                                                               mass_mean=mass_mean, mass_std=mass_std)

    # Generate samples for galaxy-hosted GW events (order: ra, dec, dL, m1det, m2det)
    ra_unc = config['gw_samples']['ra_uncertainty']
    dec_unc = config['gw_samples']['dec_uncertainty']
    mass_unc = config['gw_samples']['mass_uncertainty']
    n_initial_samples = 256000  # Fixed value
    
    ras_gal, decs_gal, dLs_gal, m1dets_gal, m2dets_gal = generate_all_samples(
        ra_gal_gw, dec_gal_gw, dL_gal_gw, m1sdet_gal_gw, m2sdet_gal_gw, N_samples_gw,
        n_initial_samples=n_initial_samples, ra_uncertainty=ra_unc, 
        dec_uncertainty=dec_unc, mass_uncertainty=mass_unc
    )

    # Generate samples for AGN-hosted GW events (order: ra, dec, dL, m1det, m2det)
    ras_agn, decs_agn, dLs_agn, m1dets_agn, m2dets_agn = generate_all_samples(
        ra_agn_gw, dec_agn_gw, dL_agn_gw, m1sdet_agn_gw, m2sdet_agn_gw, N_samples_gw,
        n_initial_samples=n_initial_samples, ra_uncertainty=ra_unc,
        dec_uncertainty=dec_unc, mass_uncertainty=mass_unc
    )

    # Keep galaxy and AGN samples separate (order: ra, dec, dL, m1det, m2det)
    # No longer shuffling to maintain consistent order with mock catalog indices
    
    # Get number of events for each type
    N_gw_gal = len(ras_gal)
    N_gw_agn = len(ras_agn)
    
    # Save samples (2D array format, separate arrays for galaxies and AGNs)
    save_samples(fn_gwsamples, ras_gal, decs_gal, dLs_gal, m1dets_gal, m2dets_gal,
                 ras_agn, decs_agn, dLs_agn, m1dets_agn, m2dets_agn, 
                 N_samples_gw, N_gw_gal, N_gw_agn)
    
    print(f"Saved {N_gw_gal} galaxy-hosted and {N_gw_agn} AGN-hosted GW events ({N_gw_gal + N_gw_agn} total) with {N_samples_gw} samples each to {fn_gwsamples}")
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    minutes = elapsed / 60
    print(f"Total time: {elapsed:.2f} s = {minutes:.2f} min")


def load_data(fn_cat, fn_gw):
    """
    Load mock catalog data and GW event indices.
    
    Parameters:
    -----------
    fn_cat : str
        Path to mock catalog HDF5 file
    fn_gw : str
        Path to GW event indices HDF5 file
        
    Returns:
    --------
    ra_gal, dec_gal, z_gal : arrays
        Galaxy positions and redshifts
    ra_agn, dec_agn, z_agn : arrays
        AGN positions and redshifts
    i_gw_gal, i_gw_agn : arrays
        Indices of galaxies/AGNs with GW events
    """
    # Load mock catalog data: positions and redshifts for galaxies and AGNs
    with h5py.File(fn_cat, 'r') as f:
        # Galaxy positions (convert from degrees to radians)
        ra_gal = np.asarray(f['ra_gal'])*np.pi/180   # Right ascension in radians
        dec_gal = np.asarray(f['dec_gal'])*np.pi/180  # Declination in radians
        z_gal = np.asarray(f['z_gal'])                # Redshift
        
        # AGN positions (convert from degrees to radians)
        ra_agn = np.asarray(f['ra_agn'])*np.pi/180   # Right ascension in radians
        dec_agn = np.asarray(f['dec_agn'])*np.pi/180  # Declination in radians
        z_agn = np.asarray(f['z_agn'])                # Redshift

    # Load indices of which galaxies/AGNs have associated GW events
    with h5py.File(fn_gw, 'r') as f:    
        i_gw_gal = np.asarray(f['i_gw_gal'])  # Indices of galaxies with GW events
        i_gw_agn = np.asarray(f['i_gw_agn'])  # Indices of AGNs with GW events
    
    return ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, i_gw_gal, i_gw_agn



def setup_cosmology(zMax_1, zMax_2, H0, Om0):
    """
    Set up redshift grids and pre-compute comoving distance lookup table.
    
    Parameters:
    -----------
    zMax_1 : float
        Low redshift boundary
    zMax_2 : float
        High redshift boundary
    H0 : float
        Hubble constant in km/s/Mpc
    Om0 : float
        Matter density parameter
        
    Returns:
    --------
    zgrid : jnp.array
        Redshift grid
    rs : jnp.array
        Comoving distance lookup table, shape (n_Om0, n_z)
    Om0grid : jnp.array
        Grid of Om0 values
    speed_of_light : float
        Speed of light in km/s
    """
    # Initialize cosmology
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    speed_of_light = constants.c.to('km/s').value  # Speed of light in km/s

    # Create redshift grid with logarithmic spacing for better resolution at low z
    # Use expm1 and log to create log-spaced grid: z = exp(log(1+z)) - 1
    zgrid_1 = np.expm1(np.linspace(np.log(1), np.log(zMax_1+1), 5000))  # Low-z grid
    zgrid_2 = np.expm1(np.linspace(np.log(zMax_1+1), np.log(zMax_2+1), 5000))  # High-z grid
    zgrid = np.concatenate([zgrid_1, zgrid_2])  # Combined redshift grid

    # Pre-compute comoving distances for a range of Om0 values
    # This creates a 2D lookup table for fast distance calculations during inference
    rs = []
    Om0grid = jnp.linspace(Om0-0.1, Om0+0.1, 100)  # Grid of Om0 values around input value
    for Om0_val in tqdm(Om0grid):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0_val)
        # Compute comoving distance for each redshift in the grid
        rs.append(cosmo.comoving_distance(zgrid).to(u.Mpc).value)

    # Convert to JAX arrays for fast interpolation
    zgrid = jnp.array(zgrid)
    rs = jnp.asarray(rs)
    rs = rs.reshape(len(Om0grid), len(zgrid))  # Shape: (n_Om0, n_z)
    
    return zgrid, rs, Om0grid, speed_of_light


def create_cosmology_functions(Om0grid, zgrid, rs, Om0, H0, speed_of_light):
    """
    Create JIT-compiled cosmological functions.
    
    Parameters:
    -----------
    Om0grid, zgrid, rs : arrays
        Cosmology lookup tables
    Om0, H0 : float
        Cosmological parameters
    speed_of_light : float
        Speed of light in km/s
        
    Returns:
    --------
    dict : Dictionary of cosmology functions
    """
    # Cosmological functions (JIT-compiled for speed)
    # These functions compute distances and related quantities in a flat Lambda-CDM universe

    @jit
    def E(z, Om0_val=Om0):
        """
        Hubble parameter normalized by H0: E(z) = H(z)/H0
        For flat Lambda-CDM: E(z) = sqrt(Om0*(1+z)^3 + (1-Om0))
        """
        return jnp.sqrt(Om0_val*(1+z)**3 + (1.0-Om0_val))

    @jit
    def r_of_z(z, H0_val, Om0_val=Om0):
        """
        Comoving distance as a function of redshift.
        Uses 2D interpolation from pre-computed lookup table, scaled by H0.
        """
        return interp2d(Om0_val, z, Om0grid, zgrid, rs)*(H0/H0_val)

    @jit
    def dL_of_z(z, H0_val, Om0_val=Om0):
        """
        Luminosity distance as a function of redshift.
        dL = (1+z) * r, where r is the comoving distance.
        """
        return (1+z)*r_of_z(z, H0_val, Om0_val)

    @jit
    def z_of_dL(dL, H0_val, Om0_val=Om0):
        """
        Inverse function: redshift as a function of luminosity distance.
        Uses interpolation to invert dL_of_z.
        """
        return jnp.interp(dL, dL_of_z(zgrid, H0_val, Om0_val), zgrid)

    @jit
    def dV_of_z(z, H0_val, Om0_val=Om0):
        """
        Comoving volume element per unit redshift: dV/dz.
        Used for computing number densities and selection functions.
        """
        return speed_of_light*r_of_z(z, H0_val, Om0_val)**2/(H0_val*E(z, Om0_val))

    @jit
    def ddL_of_z(z, dL, H0_val, Om0_val=Om0):
        """
        Derivative of luminosity distance with respect to redshift: ddL/dz.
        Used for converting between distance and redshift uncertainties.
        """
        return dL/(1+z) + speed_of_light*(1+z)/(H0_val*E(z, Om0_val))
    
    return {
        'E': E,
        'r_of_z': r_of_z,
        'dL_of_z': dL_of_z,
        'z_of_dL': z_of_dL,
        'dV_of_z': dV_of_z,
        'ddL_of_z': ddL_of_z
    }


def extract_gw_positions(ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, 
                         i_gw_gal, i_gw_agn, dL_of_z, H0):
    """
    Extract positions and compute luminosity distances for GW events.
    
    Parameters:
    -----------
    ra_gal, dec_gal, z_gal : arrays
        Galaxy positions and redshifts
    ra_agn, dec_agn, z_agn : arrays
        AGN positions and redshifts
    i_gw_gal, i_gw_agn : arrays
        Indices of galaxies/AGNs with GW events
    dL_of_z : function
        Function to compute luminosity distance from redshift
    H0 : float
        Hubble constant
        
    Returns:
    --------
    ra_gal_gw, dec_gal_gw, dL_gal_gw : arrays
        Positions and distances for galaxy-hosted GW events
    ra_agn_gw, dec_agn_gw, dL_agn_gw : arrays
        Positions and distances for AGN-hosted GW events
    """
    # Extract positions and compute luminosity distances for GW events from galaxies
    ra_gal_gw = ra_gal[i_gw_gal]      # Right ascension of galaxies with GW events
    dec_gal_gw = dec_gal[i_gw_gal]    # Declination of galaxies with GW events
    dL_gal_gw = dL_of_z(z_gal[i_gw_gal], H0)  # Luminosity distance in Mpc

    # Extract positions and compute luminosity distances for GW events from AGNs
    ra_agn_gw = ra_agn[i_gw_agn]      # Right ascension of AGNs with GW events
    dec_agn_gw = dec_agn[i_gw_agn]    # Declination of AGNs with GW events
    dL_agn_gw = dL_of_z(z_agn[i_gw_agn], H0)  # Luminosity distance in Mpc
    
    return ra_gal_gw, dec_gal_gw, dL_gal_gw, ra_agn_gw, dec_agn_gw, dL_agn_gw


def generate_black_hole_masses(z, N_gw, mass_mean=35, mass_std=5):
    """
    Generate black hole masses for GW events.
    
    Parameters:
    -----------
    z : array
        Redshifts of the events
    N_gw : int
        Number of GW events
    mass_mean : float
        Mean black hole mass in solar masses
    mass_std : float
        Standard deviation of black hole mass in solar masses
        
    Returns:
    --------
    m1sdet, m2sdet : arrays
        Detector-frame (redshifted) masses
    """
    # Generate masses from normal distribution
    m1s = np.random.normal(mass_mean, mass_std, N_gw)
    m2s = np.random.normal(mass_mean, mass_std, N_gw)
    # Sort so m1 >= m2 (convention: m1 is the more massive black hole)
    m2s_gw, m1s_gw = np.sort([m1s, m2s], axis=0)
    
    # Compute detector-frame (redshifted) masses
    # The observed mass is the source-frame mass multiplied by (1+z)
    m1sdet = m1s_gw*(1+z)
    m2sdet = m2s_gw*(1+z)
    
    return m1sdet, m2sdet


def generate_event_samples(ra, dec, dL, m1det, m2det, N_samples_gw, n_initial_samples=256000, 
                           ra_uncertainty=0.01, dec_uncertainty=0.01, mass_uncertainty=1.5):
    """
    Generate samples for a single GW event with measurement uncertainties.
    Order: ra, dec, dL, m1det, m2det
    
    Parameters:
    -----------
    ra : float
        True right ascension in radians
    dec : float
        True declination in radians
    dL : float
        True luminosity distance
    m1det : float
        True detector-frame mass 1
    m2det : float
        True detector-frame mass 2
    N_samples_gw : int
        Number of samples to generate
    n_initial_samples : int
        Number of initial samples to generate (before filtering)
    ra_uncertainty : float
        Standard deviation of RA uncertainty in radians
    dec_uncertainty : float
        Standard deviation of dec uncertainty in radians
    mass_uncertainty : float
        Standard deviation of mass uncertainty in solar masses
        
    Returns:
    --------
    ra_samples, dec_samples, dL_samples, m1det_samples, m2det_samples : arrays
        Samples in order: ra, dec, dL, m1det, m2det
    """
    # Mean vector in order: ra, dec, dL, m1det, m2det
    mean = np.array([ra, dec, dL, m1det, m2det])
    # Covariance matrix (diagonal, independent uncertainties)
    cov = np.diag([ra_uncertainty**2, dec_uncertainty**2, dL, 
                  mass_uncertainty**2, mass_uncertainty**2])
    dec_idx = 1  # Declination is at index 1 in the 5D samples
    
    rv = multivariate_normal(mean, cov)
    
    # Generate a large number of samples to ensure we have enough valid ones
    # after filtering by declination bounds
    samples = rv.rvs([n_initial_samples])
    
    # Filter samples to ensure declination is within valid range [-pi/2, pi/2]
    dec_samples = samples[:,dec_idx]
    mask = np.where((dec_samples > -np.pi/2) & (dec_samples < np.pi/2))
    samples = samples[mask]
    
    # Randomly select N_samples_gw samples from the valid ones
    choose = np.random.randint(0, len(samples), N_samples_gw)
    samples = samples[choose]

    # Return samples in order: ra, dec, dL, m1det, m2det, wrapping RA to [0, 2*pi)
    ra_samples = samples[:,0] % (2 * np.pi)
    dec_samples = samples[:,1]
    dL_samples = samples[:,2]
    m1det_samples = samples[:,3]
    m2det_samples = samples[:,4]
    
    return ra_samples, dec_samples, dL_samples, m1det_samples, m2det_samples


def generate_all_samples(ra_gw, dec_gw, dL_gw, m1det_gw, m2det_gw, N_samples_gw, 
                        n_initial_samples=256000, ra_uncertainty=0.01, 
                        dec_uncertainty=0.01, mass_uncertainty=1.5):
    """
    Generate samples for all GW events of a given type.
    Order: ra, dec, dL, m1det, m2det
    
    Parameters:
    -----------
    ra_gw, dec_gw, dL_gw : arrays
        Positions and distances for GW events
    m1det_gw, m2det_gw : arrays
        Detector-frame masses
    N_samples_gw : int
        Number of samples per event
    n_initial_samples : int
        Number of initial samples to generate (before filtering)
    ra_uncertainty : float
        Standard deviation of RA uncertainty in radians
    dec_uncertainty : float
        Standard deviation of dec uncertainty in radians
    mass_uncertainty : float
        Standard deviation of mass uncertainty in solar masses
        
    Returns:
    --------
    ras, decs, dLs, m1dets, m2dets : lists
        Lists of sample arrays, one per event, in order: ra, dec, dL, m1det, m2det
    """
    ras = []
    decs = []
    dLs = []
    m1dets = []
    m2dets = []
    
    for k in tqdm(range(len(ra_gw))):
        ra_samples, dec_samples, dL_samples, m1det_samples, m2det_samples = generate_event_samples(
            ra_gw[k], dec_gw[k], dL_gw[k], m1det_gw[k], m2det_gw[k], N_samples_gw,
            n_initial_samples=n_initial_samples, ra_uncertainty=ra_uncertainty,
            dec_uncertainty=dec_uncertainty, mass_uncertainty=mass_uncertainty
        )
        ras.append(ra_samples)
        decs.append(dec_samples)
        dLs.append(dL_samples)
        m1dets.append(m1det_samples)
        m2dets.append(m2det_samples)
    
    return ras, decs, dLs, m1dets, m2dets


def shuffle_events(ras, decs, dLs, m1dets, m2dets):
    """
    Shuffle the events to mix galaxies and AGNs together.
    Order: ra, dec, dL, m1det, m2det
    
    Parameters:
    -----------
    ras, decs, dLs, m1dets, m2dets : lists
        Lists of sample arrays
        
    Returns:
    --------
    ras, decs, dLs, m1dets, m2dets : lists
        Shuffled lists in order: ra, dec, dL, m1det, m2det
    """
    print(f"Shuffling {len(ras)} events to mix galaxies and AGNs...")
    shuffle_indices = np.random.permutation(len(ras))
    ras = [ras[i] for i in shuffle_indices]
    decs = [decs[i] for i in shuffle_indices]
    dLs = [dLs[i] for i in shuffle_indices]
    m1dets = [m1dets[i] for i in shuffle_indices]
    m2dets = [m2dets[i] for i in shuffle_indices]
    
    return ras, decs, dLs, m1dets, m2dets


def save_samples(fn_gwsamples, ras_gal, decs_gal, dLs_gal, m1dets_gal, m2dets_gal,
                 ras_agn, decs_agn, dLs_agn, m1dets_agn, m2dets_agn, N_samples_gw, N_gw_gal, N_gw_agn):
    """
    Save samples to HDF5 file as 2D arrays with shape (N_gw, N_samples_gw).
    Separate arrays for galaxies and AGNs with suffixes.
    Order: ra, dec, dL, m1det, m2det
    
    Parameters:
    -----------
    fn_gwsamples : str
        Path to output file
    ras_gal, decs_gal, dLs_gal, m1dets_gal, m2dets_gal : lists
        Lists of sample arrays for galaxies (one array per event)
    ras_agn, decs_agn, dLs_agn, m1dets_agn, m2dets_agn : lists
        Lists of sample arrays for AGNs (one array per event)
    N_samples_gw : int
        Number of samples per event
    N_gw_gal : int
        Number of galaxy-hosted GW events
    N_gw_agn : int
        Number of AGN-hosted GW events
    """
    with h5py.File(fn_gwsamples, 'w') as f:
        # Store metadata as attributes
        f.attrs['N_samples_gw'] = N_samples_gw  # Number of samples per GW event
        f.attrs['N_gw_gal'] = N_gw_gal  # Number of galaxy-hosted GW events
        f.attrs['N_gw_agn'] = N_gw_agn  # Number of AGN-hosted GW events
        f.attrs['N_gw'] = N_gw_gal + N_gw_agn  # Total number of GW events
        
        # Convert lists of arrays to 2D arrays with shape (N_gw, N_samples_gw) for galaxies
        ra_array_gal = np.array(ras_gal)  # Shape: (N_gw_gal, N_samples_gw)
        dec_array_gal = np.array(decs_gal)
        dL_array_gal = np.array(dLs_gal)
        m1det_array_gal = np.array(m1dets_gal)
        m2det_array_gal = np.array(m2dets_gal)
        
        # Convert lists of arrays to 2D arrays with shape (N_gw, N_samples_gw) for AGNs
        ra_array_agn = np.array(ras_agn)  # Shape: (N_gw_agn, N_samples_gw)
        dec_array_agn = np.array(decs_agn)
        dL_array_agn = np.array(dLs_agn)
        m1det_array_agn = np.array(m1dets_agn)
        m2det_array_agn = np.array(m2dets_agn)
        
        # Store galaxy samples as 2D arrays (compressed to save space)
        f.create_dataset('ra_gal', data=ra_array_gal, compression='gzip', shuffle=False)
        f.create_dataset('dec_gal', data=dec_array_gal, compression='gzip', shuffle=False)
        f.create_dataset('dL_gal', data=dL_array_gal, compression='gzip', shuffle=False)
        f.create_dataset('m1det_gal', data=m1det_array_gal, compression='gzip', shuffle=False)
        f.create_dataset('m2det_gal', data=m2det_array_gal, compression='gzip', shuffle=False)
        
        # Store AGN samples as 2D arrays (compressed to save space)
        f.create_dataset('ra_agn', data=ra_array_agn, compression='gzip', shuffle=False)
        f.create_dataset('dec_agn', data=dec_array_agn, compression='gzip', shuffle=False)
        f.create_dataset('dL_agn', data=dL_array_agn, compression='gzip', shuffle=False)
        f.create_dataset('m1det_agn', data=m1det_array_agn, compression='gzip', shuffle=False)
        f.create_dataset('m2det_agn', data=m2det_array_agn, compression='gzip', shuffle=False)


def load_samples(filename, N_samples_gw=None):
    """
    Load gravitational wave samples from HDF5 file.
    Returns numpy arrays as 2D arrays (not flattened). No JAX dependency.
    Returns separate arrays for galaxies and AGNs.
    
    Parameters
    ----------
    filename : str
        Path to GW samples HDF5 file
    N_samples_gw : int, optional
        Number of samples per event to use (default: None, uses all)
    
    Returns
    -------
    tuple
        (ra_gal, dec_gal, dL_gal, m1det_gal, m2det_gal, 
         ra_agn, dec_agn, dL_agn, m1det_agn, m2det_agn)
        Each is a numpy array with shape (N_gw, N_samples_gw)
    """
    print(f"Loading GW samples from {filename} (N_samples_gw={N_samples_gw})")
    with h5py.File(filename, 'r') as inp:
        # Support both new (N_samples_gw) and old (nsamp) attribute names for backward compatibility
        N_samples_gw_ = inp.attrs.get('N_samples_gw', inp.attrs.get('nsamp', None))
        if N_samples_gw_ is None:
            raise ValueError("File missing N_samples_gw or nsamp attribute")
        # Support both new (N_gw_*) and old (nobs_*) attribute names for backward compatibility
        N_gw_gal = inp.attrs.get('N_gw_gal', inp.attrs.get('nobs_gal', 0))
        N_gw_agn = inp.attrs.get('N_gw_agn', inp.attrs.get('nobs_agn', 0))
        
        # Determine slice sizes
        if N_samples_gw is None:
            N_samples_gw = N_samples_gw_
        N_samples_gw = min(N_samples_gw, N_samples_gw_)
        
        # Load galaxy samples
        if N_gw_gal > 0:
            ra_gal = np.array(inp['ra_gal'][:, 0:N_samples_gw])
            dec_gal = np.array(inp['dec_gal'][:, 0:N_samples_gw])
            dL_gal_data = inp['dL_gal'][:, 0:N_samples_gw]
            dL_gal = np.array((np.array(dL_gal_data) * u.Mpc).value)
            m1det_gal = np.array(inp['m1det_gal'][:, 0:N_samples_gw])
            m2det_gal = np.array(inp['m2det_gal'][:, 0:N_samples_gw])
        else:
            # Handle edge case when there are no galaxy events
            ra_gal = np.empty((0, N_samples_gw))
            dec_gal = np.empty((0, N_samples_gw))
            dL_gal = np.empty((0, N_samples_gw))
            m1det_gal = np.empty((0, N_samples_gw))
            m2det_gal = np.empty((0, N_samples_gw))
        
        # Load AGN samples (can be empty when f_agn=0)
        if N_gw_agn > 0:
            ra_agn = np.array(inp['ra_agn'][:, 0:N_samples_gw])
            dec_agn = np.array(inp['dec_agn'][:, 0:N_samples_gw])
            dL_agn_data = inp['dL_agn'][:, 0:N_samples_gw]
            dL_agn = np.array((np.array(dL_agn_data) * u.Mpc).value)
            m1det_agn = np.array(inp['m1det_agn'][:, 0:N_samples_gw])
            m2det_agn = np.array(inp['m2det_agn'][:, 0:N_samples_gw])
        else:
            # Avoid slicing 1D datasets when there are zero AGN events
            ra_agn = np.empty((0, N_samples_gw))
            dec_agn = np.empty((0, N_samples_gw))
            dL_agn = np.empty((0, N_samples_gw))
            m1det_agn = np.empty((0, N_samples_gw))
            m2det_agn = np.empty((0, N_samples_gw))
    
    return ra_gal, dec_gal, dL_gal, m1det_gal, m2det_gal, ra_agn, dec_agn, dL_agn, m1det_agn, m2det_agn



if __name__ == '__main__':
    config, overwrite = parse_args()
    main(config, overwrite)
