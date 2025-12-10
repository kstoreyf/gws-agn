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
if '--undefok' not in ' '.join(sys.argv):
    sys.argv.insert(1, '--undefok=coordination_agent_recoverable')

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
from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
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

# Parameters for the mock catalog
fagn = 0.5  # Fraction of AGNs (not used in this script but part of filename)
lam = 0.5   # Lambda parameter for AGNs (not used in this script but part of filename)

# File paths for input data
filepath = '../data/mocks_glass/mock_seed42_ratioNgalNagn1_bgal1.0_bagn1.0/'
mockpath = filepath + 'mock_catalog.h5'  # Mock catalog with galaxy and AGN positions
mockgwpath = filepath + 'gws_fagn'+str(fagn)+'_lambdaagn'+str(lam)+'_N1000_seed1042.h5'  # GW event indices

# Load mock catalog data: positions and redshifts for galaxies and AGNs
with h5py.File(mockpath, 'r') as f:
    # Galaxy positions (convert from degrees to radians)
    ra_gal = np.asarray(f['ra_gal'])*np.pi/180   # Right ascension in radians
    dec_gal = np.asarray(f['dec_gal'])*np.pi/180  # Declination in radians
    z_gal = np.asarray(f['z_gal'])                # Redshift
    
    # AGN positions (convert from degrees to radians)
    ra_agn = np.asarray(f['ra_agn'])*np.pi/180   # Right ascension in radians
    dec_agn = np.asarray(f['dec_agn'])*np.pi/180  # Declination in radians
    z_agn = np.asarray(f['z_agn'])                # Redshift

# Load indices of which galaxies/AGNs have associated GW events
with h5py.File(mockgwpath, 'r') as f:    
    i_gw_gal = np.asarray(f['i_gw_gal'])  # Indices of galaxies with GW events
    i_gw_agn = np.asarray(f['i_gw_agn'])  # Indices of AGNs with GW events
    
# Number of samples to generate per GW event
nsamp = 10000
# Total number of observed GW events (from both galaxies and AGNs)
nobs = len(i_gw_agn) + len(i_gw_gal)

# Redshift grid boundaries for cosmology calculations
zMax_1 = 0.5  # Low redshift boundary
zMax_2 = 5    # High redshift boundary

# Cosmological parameters from Planck 2015
H0Planck = Planck15.H0.value  # Hubble constant in km/s/Mpc
Om0Planck = Planck15.Om0      # Matter density parameter

# Initialize cosmology with Planck parameters
cosmo = FlatLambdaCDM(H0=H0Planck,Om0=Planck15.Om0)
speed_of_light = constants.c.to('km/s').value  # Speed of light in km/s

# Create redshift grid with logarithmic spacing for better resolution at low z
# Use expm1 and log to create log-spaced grid: z = exp(log(1+z)) - 1
zgrid_1 = np.expm1(np.linspace(np.log(1), np.log(zMax_1+1), 5000))  # Low-z grid
zgrid_2 = np.expm1(np.linspace(np.log(zMax_1+1), np.log(zMax_2+1), 5000))  # High-z grid
zgrid = np.concatenate([zgrid_1,zgrid_2])  # Combined redshift grid

# Pre-compute comoving distances for a range of Om0 values
# This creates a 2D lookup table for fast distance calculations during inference
rs = []
Om0grid = jnp.linspace(Om0Planck-0.1, Om0Planck+0.1, 100)  # Grid of Om0 values around Planck value
for Om0 in tqdm(Om0grid):
    cosmo = FlatLambdaCDM(H0=H0Planck, Om0=Om0)
    # Compute comoving distance for each redshift in the grid
    rs.append(cosmo.comoving_distance(zgrid).to(u.Mpc).value)

# Convert to JAX arrays for fast interpolation
zgrid = jnp.array(zgrid)
rs = jnp.asarray(rs)
rs = rs.reshape(len(Om0grid), len(zgrid))  # Shape: (n_Om0, n_z)

# Cosmological functions (JIT-compiled for speed)
# These functions compute distances and related quantities in a flat Lambda-CDM universe

@jit
def E(z, Om0=Om0Planck):
    """
    Hubble parameter normalized by H0: E(z) = H(z)/H0
    For flat Lambda-CDM: E(z) = sqrt(Om0*(1+z)^3 + (1-Om0))
    """
    return jnp.sqrt(Om0*(1+z)**3 + (1.0-Om0))

@jit
def r_of_z(z, H0, Om0=Om0Planck):
    """
    Comoving distance as a function of redshift.
    Uses 2D interpolation from pre-computed lookup table, scaled by H0.
    """
    return interp2d(Om0, z, Om0grid, zgrid, rs)*(H0Planck/H0)

@jit
def dL_of_z(z, H0, Om0=Om0Planck):
    """
    Luminosity distance as a function of redshift.
    dL = (1+z) * r, where r is the comoving distance.
    """
    return (1+z)*r_of_z(z, H0, Om0)

@jit
def z_of_dL(dL, H0, Om0=Om0Planck):
    """
    Inverse function: redshift as a function of luminosity distance.
    Uses interpolation to invert dL_of_z.
    """
    return jnp.interp(dL, dL_of_z(zgrid, H0, Om0), zgrid)

@jit
def dV_of_z(z, H0, Om0=Om0Planck):
    """
    Comoving volume element per unit redshift: dV/dz.
    Used for computing number densities and selection functions.
    """
    return speed_of_light*r_of_z(z, H0, Om0)**2/(H0*E(z, Om0))

@jit
def ddL_of_z(z, dL, H0, Om0=Om0Planck):
    """
    Derivative of luminosity distance with respect to redshift: ddL/dz.
    Used for converting between distance and redshift uncertainties.
    """
    return dL/(1+z) + speed_of_light*(1+z)/(H0*E(z, Om0))

# Extract positions and compute luminosity distances for GW events from galaxies
ra_gal_gw = ra_gal[i_gw_gal]      # Right ascension of galaxies with GW events
dec_gal_gw = dec_gal[i_gw_gal]    # Declination of galaxies with GW events
dL_gal_gw = dL_of_z(z_gal[i_gw_gal], H0Planck)  # Luminosity distance in Mpc

# Extract positions and compute luminosity distances for GW events from AGNs
ra_agn_gw = ra_agn[i_gw_agn]      # Right ascension of AGNs with GW events
dec_agn_gw = dec_agn[i_gw_agn]    # Declination of AGNs with GW events
dL_agn_gw = dL_of_z(z_agn[i_gw_agn], H0Planck)  # Luminosity distance in Mpc

# Generate black hole masses for galaxy-hosted GW events
# Masses are drawn from a normal distribution (mean=35 M_sun, std=5 M_sun)
m1s = np.random.normal(35, 5, len(i_gw_gal))
m2s = np.random.normal(35, 5, len(i_gw_gal))
# Sort so m1 >= m2 (convention: m1 is the more massive black hole)
m2s_gal_gw, m1s_gal_gw = np.sort([m1s, m2s], axis=0)

# Generate black hole masses for AGN-hosted GW events
m1s = np.random.normal(35, 5, len(i_gw_agn))
m2s = np.random.normal(35, 5, len(i_gw_agn))
m2s_agn_gw, m1s_agn_gw = np.sort([m1s, m2s], axis=0)

# Compute detector-frame (redshifted) masses for galaxy-hosted events
# The observed mass is the source-frame mass multiplied by (1+z)
m1sdet_gal_gw = m1s_gal_gw*(1+z_gal[i_gw_gal])
m2sdet_gal_gw = m2s_gal_gw*(1+z_gal[i_gw_gal])

# Compute detector-frame (redshifted) masses for AGN-hosted events
m1sdet_agn_gw = m1s_agn_gw*(1+z_agn[i_gw_agn])
m2sdet_agn_gw = m2s_agn_gw*(1+z_agn[i_gw_agn])

# Lists to store samples for all GW events
dLs = []  # Luminosity distance samples
ras = []  # Right ascension samples
decs = [] # Declination samples

# Generate samples for GW events from galaxies
for k in tqdm(range(int(len(i_gw_gal)))):
    # True values for this GW event
    dL = dL_gal_gw[k]  # True luminosity distance
    ra = ra_gal_gw[k]   # True right ascension
    dec = dec_gal_gw[k] # True declination
    mean = np.array([dL, ra, dec])
    
    # Covariance matrix for the multivariate normal distribution
    # Distance uncertainty scales with distance (dL), position uncertainties are fixed
    cov = np.diag([dL, 0.01**2, 0.01**2])  # [dL uncertainty, ra uncertainty, dec uncertainty]
    rv = multivariate_normal(mean, cov)
    
    # Generate a large number of samples (256000) to ensure we have enough valid ones
    # after filtering by declination bounds
    samples = rv.rvs([256000])
    
    # Filter samples to ensure declination is within valid range [-pi/2, pi/2]
    dec_samples = samples[:,2]
    mask = np.where((dec_samples > -np.pi/2) & (dec_samples < np.pi/2))
    samples = samples[mask]
    
    # Randomly select nsamp samples from the valid ones
    choose = np.random.randint(0, len(samples), nsamp)
    samples = samples[choose]

    # Store samples, wrapping RA to [0, 2*pi)
    dLs.append(samples[:,0])
    ras.append(samples[:,1] % (2 * np.pi))
    decs.append(samples[:,2])

# Generate samples for GW events from AGNs (same procedure as for galaxies)
for k in tqdm(range(int(len(i_gw_agn)))):
    # True values for this GW event
    dL = dL_agn_gw[k]  # True luminosity distance
    ra = ra_agn_gw[k]   # True right ascension
    dec = dec_agn_gw[k] # True declination
    mean = np.array([dL, ra, dec])
    
    # Covariance matrix: distance uncertainty scales with distance, position uncertainties fixed
    cov = np.diag([dL, 0.01**2, 0.01**2])
    rv = multivariate_normal(mean, cov)
    
    # Generate many samples to ensure enough valid ones after filtering
    samples = rv.rvs([256000])
    
    # Filter to valid declination range [-pi/2, pi/2]
    dec_samples = samples[:,2]
    mask = np.where((dec_samples > -np.pi/2) & (dec_samples < np.pi/2))
    samples = samples[mask]
    
    # Randomly select nsamp samples
    choose = np.random.randint(0, len(samples), nsamp)
    samples = samples[choose]
    
    # Store samples, wrapping RA to [0, 2*pi)
    dLs.append(samples[:,0])
    ras.append(samples[:,1] % (2 * np.pi))
    decs.append(samples[:,2])

# Shuffle the events to mix galaxies and AGNs together
# This ensures that taking the first N events gives a representative random subsample
print(f"Shuffling {len(dLs)} events to mix galaxies and AGNs...")
shuffle_indices = np.random.permutation(len(dLs))
dLs = [dLs[i] for i in shuffle_indices]
ras = [ras[i] for i in shuffle_indices]
decs = [decs[i] for i in shuffle_indices]

# Save all samples to HDF5 file for later use in inference
output_filename = f'{filepath}gwsamples_fagn{str(fagn)}_lambdaagn{str(lam)}_N1000_seed1042_pos_only.h5'
with h5py.File(output_filename, 'w') as f:
    # Store metadata as attributes
    f.attrs['nsamp'] = nsamp  # Number of samples per GW event
    f.attrs['nobs'] = nobs     # Total number of GW events
    
    # Store samples as datasets (compressed to save space)
    # Each dataset is a list of arrays, one array per GW event
    f.create_dataset('dL', data=dLs, compression='gzip', shuffle=False)   # Luminosity distances
    f.create_dataset('ra', data=ras, compression='gzip', shuffle=False)  # Right ascensions
    f.create_dataset('dec', data=decs, compression='gzip', shuffle=False) # Declinations

