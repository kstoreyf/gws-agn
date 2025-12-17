"""
Dark siren inference pipeline for GW-AGN analysis.

This module performs Bayesian inference on gravitational wave dark sirens
using galaxy and AGN catalogs to constrain cosmological parameters.

run with glassenv (has jax properly installed; gwsagn env doesn't)
"""

import os
import sys

# Set environment variables BEFORE any imports
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#ksf adding this
# os.environ['JAX_PLATFORMS'] = 'cpu'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings


# Fix for coordination_agent_recoverable flag conflict
# Add --undefok flag to sys.argv BEFORE any absl imports
# if '--undefok' not in ' '.join(sys.argv):
#     sys.argv.insert(1, '--undefok=coordination_agent_recoverable')

# Parse absl flags early with undefok to prevent redefinition errors
try:
    from absl import flags
    # Parse with undefok before any TensorFlow/JAX initialization
    flags.FLAGS.parse_flags_with_usage(sys.argv)
except (Exception, SystemExit):
    # Ignore flag parsing errors at this stage
    pass

import warnings
warnings.filterwarnings('ignore')

import jax

from jax import random, jit, vmap, grad
from jax import numpy as jnp
from jax.lax import cond
from jax.scipy.special import logsumexp


import numpy as np
import healpy as hp
import h5py
import astropy.units as u
from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
import astropy.constants as constants
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from jax.scipy.stats import norm
from tqdm import tqdm
import json
from datetime import datetime

from jaxinterp2d import interp2d, CartesianGrid

import time
import argparse
import yaml

import utils

try:
    import emcee
except ImportError:
    emcee = None



def parse_args():
    """
    Parse command line arguments for config file.
    
    Returns:
    --------
    config : dict
        Configuration dictionary for inference (from YAML file)
    overwrite : bool
        Whether to overwrite existing output files
    """
    parser = argparse.ArgumentParser(description='Run dark siren inference')
    parser.add_argument('config', type=str, nargs='?', help='Path to YAML configuration file for inference')
    parser.add_argument('--config', dest='config_flag', type=str,
                        help='Path to YAML configuration file for inference (legacy flag)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing output file if it exists (default: False)')
    args = parser.parse_args()
    
    # Prefer positional config, fall back to --config for backwards compatibility
    config_path = args.config or args.config_flag
    if not config_path:
        parser.error('Please provide the config file as the first argument or via --config.')

    # Load inference config file
    with open(config_path, 'r') as f:
        config_inference = yaml.safe_load(f)
    
    # Load data config file referenced in inference config
    fn_config_data = config_inference['fn_config_data']
    
    with open(fn_config_data, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Get absolute path to inference config file
    fn_config = os.path.abspath(config_path)
    
    return config_inference, config_data, fn_config, args.overwrite


def main(config_inference, config_data, fn_config, overwrite=False):
    """
    Main function to run the inference pipeline.
    
    Parameters
    ----------
    config_data : dict
        Configuration dictionary for data (from YAML file)
    config_inference : dict
        Configuration dictionary for inference (from YAML file)
    fn_config : str
        Path to inference config file
    overwrite : bool
        Whether to overwrite existing output files (default: False)
    Returns
    -------
    dict
        Dictionary containing results depending on mode_inf:
        - For 'mcmc': posterior_samples, sampler, config, mcmc_params
        - For 'grid': log_likelihood_grid, H0_grid, alpha_agn_grid, config
    """
    import time
    t_start = time.perf_counter()
    
    # Extract parameters from config_data
    nside = config_data['pixelization']['nside']

    # Extract parameters from config_inference
    mode_inf = config_inference['mode_inf']
    
    # MCMC parameters
    N_walkers = config_inference['mcmc']['N_walkers']
    N_steps = config_inference['mcmc']['N_steps']
    burnin_frac = config_inference['mcmc']['burnin_frac']
    seed_mcmc = config_inference['mcmc']['seed_mcmc']
    
    # Likelihood grid parameters
    N_H0 = config_inference['grid']['N_H0']
    N_alpha_agn = config_inference['grid']['N_alpha_agn']
    
    # Parameter bounds (set directly, not from config)
    H0_bounds = (50, 100)
    alpha_agn_bounds = (0, 1)
    Om0_bounds = None
    gamma_agn_bounds = (-5, 5)
    gamma_gal_bounds = (-5, 5)
    
    # Cosmology parameters
    Om0 = config_inference['parameters']['Om0']
    gamma_agn = config_inference['parameters']['gamma_agn']
    gamma_gal = config_inference['parameters']['gamma_gal']
    
    # Number of GW events to use in inference
    N_gw_inf = config_inference.get('N_gw_inf', None)
    
    # Output settings - get filename from config
    fn_inf = config_inference['paths']['fn_inf']
    
    # Check if output file exists and handle overwrite flag
    if fn_inf is not None and os.path.exists(fn_inf):
        if not overwrite:
            raise FileExistsError(
                f"Output file {fn_inf} already exists. Use --overwrite to overwrite it."
            )
        else:
            print(f"Warning: Output file {fn_inf} exists and will be overwritten (--overwrite flag set)")

    print(f"Running main inference pipeline: mode_inf={mode_inf}, fn_config={fn_config}, \
            N_gw_inf={N_gw_inf}")
    
    # Construct file paths using new naming convention
    dir_mock = config_data['paths']['dir_mock']
    fn_cat_gal_pixelated = os.path.join(dir_mock, config_data['paths']['name_cat_gal_pixelated'])
    fn_cat_agn_pixelated = os.path.join(dir_mock, config_data['paths']['name_cat_agn_pixelated'])
    fn_gwsamples = os.path.join(dir_mock, config_data['paths']['name_gwsamples'])

    #Check JAX GPU status
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"GPU devices: {[d for d in jax.devices() if d.device_kind == 'gpu']}")
    #print(f"Test array device: {jnp.array([1.0]).device()}")

    # Load catalog data
    catalog_data = load_catalog_data(fn_cat_gal_pixelated, fn_cat_agn_pixelated, nside=nside)
    
    # Setup cosmology
    cosmo_funcs = setup_cosmology()
    
    # Create catalog probability functions
    prob_funcs = create_catalog_probability_functions(catalog_data)

    # Load GW samples
    gw_data = load_gw_samples(fn_gwsamples, N_gw_inf=N_gw_inf)

    # Set up MCMC parameters
    mcmc_params = setup_mcmc_parameters(H0_bounds=H0_bounds, alpha_agn_bounds=alpha_agn_bounds, Om0_bounds=Om0_bounds, 
                                        gamma_agn_bounds=gamma_agn_bounds, gamma_gal_bounds=gamma_gal_bounds)

    # Run inference based on mode_inf
    if mode_inf == 'grid':
        results = run_likelihood_grid(
            gw_data, catalog_data, cosmo_funcs, prob_funcs, mcmc_params,
            config_inference, fn_config,
            Om0=Om0, gamma_agn=gamma_agn, gamma_gal=gamma_gal,
            N_H0=N_H0, N_alpha_agn=N_alpha_agn,
            fn_inf=fn_inf
        )
    elif mode_inf == 'mcmc':
        results = run_inference_mcmc(
            gw_data, catalog_data, cosmo_funcs, prob_funcs, mcmc_params,
            config_inference, fn_config,
            N_walkers=N_walkers, N_steps=N_steps, burnin_frac=burnin_frac,
            Om0=Om0, gamma_agn=gamma_agn, gamma_gal=gamma_gal,
            seed=seed_mcmc, fn_inf=fn_inf
        )
    else:
        raise ValueError(f"Unknown mode_inf: {mode_inf}. Must be 'mcmc' or 'grid'")
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    minutes = elapsed / 60
    print(f"Total time: {elapsed:.2f} s = {minutes:.2f} min")
    
    return results


def run_likelihood_grid(
    gw_data, catalog_data, cosmo_funcs, prob_funcs, mcmc_params,
    config_inference, fn_config,
    Om0=None, gamma_agn=0, gamma_gal=0,
    N_H0=30, N_alpha_agn=30,
    fn_inf=None
):
    """
    Run likelihood grid computation.
    
    Parameters
    ----------
    gw_data : dict
        Dictionary from load_gw_samples()
    catalog_data : dict
        Dictionary from load_catalog_data()
    cosmo_funcs : dict
        Dictionary from setup_cosmology()
    prob_funcs : dict
        Dictionary from create_catalog_probability_functions()
    mcmc_params : dict
        Dictionary from setup_mcmc_parameters()
    Om0 : float, optional
        Matter density parameter (default: None, uses Planck value)
    gamma_agn : float
        AGN evolution parameter (default: 0)
    gamma_gal : float
        Galaxy evolution parameter (default: 0)
    N_H0 : int
        Number of H0 grid points (default: 50)
    N_alpha_agn : int
        Number of alpha_agn grid points (default: 50)
    fn_inf : str, optional
        Path to save likelihood grid (default: None)
    
    Returns
    -------
    dict
        Dictionary containing:
        - log_likelihood_grid: 2D array of log-likelihood values
        - H0_grid: 1D array of H0 values
        - alpha_agn_grid: 1D array of alpha_agn values
        - config: configuration dictionary
    """
    print(f"Running likelihood grid computation: N_H0={N_H0}, N_alpha_agn={N_alpha_agn}")
    
    # Create parameter grids from bounds
    H0_grid = np.linspace(mcmc_params['lower_bound'][0], mcmc_params['upper_bound'][0], N_H0)
    alpha_agn_grid = np.linspace(mcmc_params['lower_bound'][1], mcmc_params['upper_bound'][1], N_alpha_agn)
    
    # Compute likelihood grid
    log_likelihood_grid = compute_likelihood_grid(
        gw_data, catalog_data, cosmo_funcs, prob_funcs,
        H0_grid, alpha_agn_grid,
        Om0=Om0, gamma_agn=gamma_agn, gamma_gal=gamma_gal,
        progress=True
    )
    
    # Use config_inference directly (read-only, no modifications)
    
    # Save results if output file specified
    if fn_inf is not None:
        save_likelihood_grid(
            fn_inf,
            log_likelihood_grid, H0_grid, alpha_agn_grid,
            fn_config=fn_config,
            grid_params={'Om0': Om0, 'gamma_agn': gamma_agn, 'gamma_gal': gamma_gal}
        )
        print(f"Likelihood grid saved to {fn_inf}")
    
    return {
        'log_likelihood_grid': log_likelihood_grid,
        'H0_grid': H0_grid,
        'alpha_agn_grid': alpha_agn_grid,
        'config': config_inference,
        'mcmc_params': mcmc_params
    }


def run_inference_mcmc(
    gw_data, catalog_data, cosmo_funcs, prob_funcs, mcmc_params,
    config_inference, fn_config,
    N_walkers=16, N_steps=1000, burnin_frac=0.2,
    Om0=None, gamma_agn=0, gamma_gal=0,
    seed=None, fn_inf=None
):
    """
    Run MCMC inference pipeline.
    
    Parameters
    ----------
    gw_data : dict
        Dictionary from load_gw_samples()
    catalog_data : dict
        Dictionary from load_catalog_data()
    cosmo_funcs : dict
        Dictionary from setup_cosmology()
    prob_funcs : dict
        Dictionary from create_catalog_probability_functions()
    mcmc_params : dict
        Dictionary from setup_mcmc_parameters()
    N_walkers : int
        Number of MCMC walkers (default: 16)
    N_steps : int
        Number of MCMC steps (default: 1000)
    burnin_frac : float
        Burn-in fraction (default: 0.2)
    Om0 : float, optional
        Matter density parameter (default: None, uses Planck value)
    gamma_agn : float
        AGN evolution parameter (default: 0)
    gamma_gal : float
        Galaxy evolution parameter (default: 0)
    seed : int, optional
        Random seed for MCMC initialization (default: None)
    fn_inf : str, optional
        Path to save inference results (default: None)
    
    Returns
    -------
    dict
        Dictionary containing:
        - posterior_samples: posterior samples array
        - sampler: emcee sampler object
        - config: configuration dictionary
        - mcmc_params: MCMC parameter dictionary
    """
    print(f"Running MCMC inference: N_walkers={N_walkers}, N_steps={N_steps}, burnin_frac={burnin_frac}")
    
    # Create MCMC likelihood function
    likelihood_func = create_mcmc_likelihood_function(
        gw_data, catalog_data, cosmo_funcs, prob_funcs,
        mcmc_params['lower_bound'], mcmc_params['upper_bound'],
        Om0=Om0, gamma_agn=gamma_agn, gamma_gal=gamma_gal
    )

    # Run MCMC sampling
    sampler = run_mcmc_sampling(
        likelihood_func,
        mcmc_params['lower_bound'],
        mcmc_params['upper_bound'],
        N_walkers=N_walkers,
        N_steps=N_steps,
        seed=seed
    )

    # Get posterior samples
    posterior_samples = get_posterior_samples(sampler, burnin_frac=burnin_frac)
    
    # Use config_inference directly (read-only, no modifications)
    
    # Save results if output file specified
    if fn_inf is not None:
        save_inference_results(
            fn_inf,
            posterior_samples,
            sampler=sampler,
            mcmc_params=mcmc_params,
            fn_config=fn_config
        )
        print(f"Inference results saved to {fn_inf}")
    
    return {
        'posterior_samples': posterior_samples,
        'sampler': sampler,
        'config': config_inference,
        'mcmc_params': mcmc_params
    }


def save_inference_results(
    fn_inf, posterior_samples, sampler=None, mcmc_params=None,
    fn_config=None
):
    """
    Save inference results to HDF5 file.
    
    Parameters
    ----------
    fn_inf : str
        Output HDF5 filename for inference results
    posterior_samples : array
        Posterior samples array (N_samples, N_params)
    sampler : emcee.EnsembleSampler, optional
        Full sampler object (saves full chain if provided)
    mcmc_params : dict, optional
        MCMC parameter dictionary from setup_mcmc_parameters()
    fn_config : str, optional
        Path to inference config file
    """
    print(f"Saving inference results to {fn_inf} (N_samples={len(posterior_samples)})")
    os.makedirs(os.path.dirname(fn_inf), exist_ok=True)
    with h5py.File(fn_inf, 'w') as f:
        # Save posterior samples
        f.create_dataset('posterior_samples', data=np.array(posterior_samples))
        
        # Save full chain if sampler provided
        if sampler is not None:
            f.create_dataset('mcmc_chain', data=np.array(sampler.chain))
            f.create_dataset('mcmc_log_prob', data=np.array(sampler.lnprobability))
            f.attrs['N_walkers'] = sampler.nwalkers
            f.attrs['N_steps'] = sampler.iteration
        
        # Save MCMC parameters
        if mcmc_params is not None:
            f.create_dataset('lower_bound', data=np.array(mcmc_params['lower_bound']))
            f.create_dataset('upper_bound', data=np.array(mcmc_params['upper_bound']))
            f.attrs['ndims'] = mcmc_params['ndims']
            # Save labels as JSON string (since HDF5 doesn't support lists of strings well)
            if 'labels' in mcmc_params:
                f.attrs['labels'] = json.dumps(mcmc_params['labels'])
        
        # Save config file path
        if fn_config is not None:
            f.attrs['fn_config'] = fn_config
        
        # Save timestamp
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['N_samples'] = len(posterior_samples)
        f.attrs['N_params'] = posterior_samples.shape[1] if len(posterior_samples.shape) > 1 else 1


def load_inference_results(fn_inf):
    """
    Load inference results from HDF5 file.
    
    Parameters
    ----------
    fn_inf : str
        Input HDF5 filename for inference results
    
    Returns
    -------
    dict
        Dictionary containing:
        - posterior_samples: array of posterior samples
        - mcmc_chain: full MCMC chain (if available)
        - mcmc_log_prob: log probabilities (if available)
        - mcmc_params: MCMC parameter dictionary
        - fn_config: path to inference config file
        - timestamp: timestamp string
        - N_samples: number of samples
        - N_params: number of parameters
    """
    print(f"Loading inference results from {fn_inf}")
    results = {}
    
    with h5py.File(fn_inf, 'r') as f:
        # Load posterior samples
        print(f.keys())
        results['posterior_samples'] = np.array(f['posterior_samples'])
        
        # Load full chain if available
        if 'mcmc_chain' in f:
            results['mcmc_chain'] = np.array(f['mcmc_chain'])
            results['mcmc_log_prob'] = np.array(f['mcmc_log_prob'])
        results['N_walkers'] = f.attrs.get('N_walkers', f.attrs.get('n_walkers', None))
        results['N_steps'] = f.attrs.get('N_steps', f.attrs.get('n_steps', None))
        
        # Load MCMC parameters
        if 'lower_bound' in f:
            mcmc_params = {
                'lower_bound': list(f['lower_bound'][:]),
                'upper_bound': list(f['upper_bound'][:]),
                'ndims': f.attrs.get('ndims', None)
            }
            if 'labels' in f.attrs:
                mcmc_params['labels'] = json.loads(f.attrs['labels'])
            results['mcmc_params'] = mcmc_params
        
        # Load attributes
        results['fn_config'] = f.attrs.get('fn_config', None)
        results['timestamp'] = f.attrs.get('timestamp', None)
        results['N_samples'] = f.attrs.get('N_samples', f.attrs.get('n_samples', None))
        results['N_params'] = f.attrs.get('N_params', f.attrs.get('n_params', None))
    
    return results


def print_inference_summary(results):
    """
    Print a summary of loaded inference results.
    
    Parameters
    ----------
    results : dict
        Dictionary from load_inference_results()
    """
    print("Printing inference results summary")
    print("=" * 60)
    print("Inference Results Summary")
    print("=" * 60)
    
    if 'timestamp' in results and results['timestamp']:
        print(f"Timestamp: {results['timestamp']}")
    
    if 'N_samples' in results and results['N_samples']:
        print(f"Number of samples: {results['N_samples']}")
    elif 'n_samples' in results and results['n_samples']:
        print(f"Number of samples: {results['n_samples']}")
    
    if 'N_params' in results and results['N_params']:
        print(f"Number of parameters: {results['N_params']}")
    elif 'n_params' in results and results['n_params']:
        print(f"Number of parameters: {results['n_params']}")
    
    if 'mcmc_params' in results:
        print("\nMCMC Parameters:")
        mcmc = results['mcmc_params']
        if 'labels' in mcmc:
            for i, label in enumerate(mcmc['labels']):
                lower = mcmc['lower_bound'][i]
                upper = mcmc['upper_bound'][i]
                print(f"  {label}: [{lower}, {upper}]")
    
    if 'fn_config' in results and results['fn_config']:
        print(f"\nConfig file: {results['fn_config']}")
    
    if 'posterior_samples' in results:
        samples = results['posterior_samples']
        print(f"\nPosterior Samples Shape: {samples.shape}")
        if len(samples) > 0:
            print("Parameter means:")
            for i in range(samples.shape[1]):
                mean = np.mean(samples[:, i])
                std = np.std(samples[:, i])
                label = f"param_{i}"
                if 'mcmc_params' in results and 'labels' in results['mcmc_params']:
                    if i < len(results['mcmc_params']['labels']):
                        label = results['mcmc_params']['labels'][i]
                print(f"  {label}: {mean:.4f} ± {std:.4f}")
    
    print("=" * 60)


def load_catalog_data(fn_cat_gal_pixelated, fn_cat_agn_pixelated, nside=None):
    """
    Load galaxy and AGN catalog data from HDF5 files.
    
    Parameters
    ----------
    fn_cat_gal_pixelated : str
        Path to galaxy catalog HDF5 file
    fn_cat_agn_pixelated : str
        Path to AGN catalog HDF5 file
    nside : int, optional
        Healpix nside parameter. If None, computed from number of pixels.
        If provided, verified against number of pixels.
    
    Returns
    -------
    dict
        Dictionary containing:
        - zgals: galaxy redshifts
        - dzgals: galaxy redshift uncertainties
        - wgals: galaxy weights
        - ngals: galaxy counts
        - zagns: AGN redshifts
        - dzagns: AGN redshift uncertainties
        - wagns: AGN weights
        - nagns: AGN counts
        - nside: healpix nside
    """
    print(f"Loading catalog data: fn_cat_gal_pixelated={fn_cat_gal_pixelated}, fn_cat_agn_pixelated={fn_cat_agn_pixelated}, nside={nside}")
    with h5py.File(fn_cat_gal_pixelated, 'r') as f:
        # Use 'z' and 'n_in_pixel' dataset names (as saved by pixelize_catalogs.py)
        zgals = jnp.asarray(f['z'])
        dzgals = 0.0001 * (1 + zgals)
        wgals = jnp.ones(zgals.shape)
        ngals = jnp.asarray(f['n_in_pixel'])
    
    with h5py.File(fn_cat_agn_pixelated, 'r') as f:
        # Use 'z' and 'n_in_pixel' dataset names (as saved by pixelize_catalogs.py)
        zagns = jnp.asarray(f['z'])
        dzagns = 0.0001 * (1 + zagns)
        wagns = jnp.ones(zagns.shape)
        nagns = jnp.asarray(f['n_in_pixel'])
    
    npix = len(zgals)
    assert npix == len(zagns), "Number of galaxies and AGNs do not match"
    
    # Handle nside: compute if None, verify if provided
    if nside is None:
        # Convert number of pixels to nside: npix = 12 * nside^2
        nside = hp.pixelfunc.npix2nside(npix)
        print(f"Computed nside={nside} from number of pixels ({npix})")
    else:
        # Verify that nside matches the number of pixels: npix = 12 * nside^2
        npix_expected = hp.pixelfunc.nside2npix(nside)
        assert npix == npix_expected, f"Number of pixels ({npix}) does not match expected for nside={nside} (expected {npix_expected})"
        print(f"Verified nside={nside} matches number of pixels ({npix})")

    print("Loaded!")
    return {
        'zgals': zgals,
        'dzgals': dzgals,
        'wgals': wgals,
        'ngals': ngals,
        'zagns': zagns,
        'dzagns': dzagns,
        'wagns': wagns,
        'nagns': nagns,
        'nside': nside
    }


def setup_cosmology(zMax_1=0.5, zMax_2=5, Om0_range=0.1, n_Om0=100):
    """
    Set up cosmology functions for redshift-distance conversions.
    
    Parameters
    ----------
    zMax_1 : float
        Maximum redshift for fine grid (default: 0.5)
    zMax_2 : float
        Maximum redshift for coarse grid (default: 5)
    Om0_range : float
        Range around Planck Om0 for interpolation grid (default: 0.1)
    n_Om0 : int
        Number of Om0 grid points (default: 100)
    
    Returns
    -------
    dict
        Dictionary containing cosmology functions and grids:
        - zgrid: redshift grid
        - Om0grid: Om0 grid
        - rs: comoving distance grid
        - E: Hubble parameter function
        - r_of_z: comoving distance function
        - dL_of_z: luminosity distance function
        - z_of_dL: redshift from luminosity distance function
        - dV_of_z: comoving volume element function
        - ddL_of_z: derivative of luminosity distance function
    """
    print(f"Setting up cosmology: zMax_1={zMax_1}, zMax_2={zMax_2}, Om0_range={Om0_range}, n_Om0={n_Om0}")
    H0_fiducial = Planck15.H0.value
    Om0_fiducial = Planck15.Om0
    speed_of_light = constants.c.to('km/s').value
    
    # Create redshift grid
    zgrid_1 = np.expm1(np.linspace(np.log(1), np.log(zMax_1 + 1), 5000))
    zgrid_2 = np.expm1(np.linspace(np.log(zMax_1 + 1), np.log(zMax_2 + 1), 1000))
    zgrid = np.concatenate([zgrid_1, zgrid_2])
    
    # Create Om0 grid and compute comoving distances
    Om0grid = jnp.linspace(Om0_fiducial - Om0_range, Om0_fiducial + Om0_range, n_Om0)
    rs = []
    for Om0 in tqdm(Om0grid):
        cosmo = FlatLambdaCDM(H0=H0_fiducial, Om0=Om0)
        rs.append(cosmo.comoving_distance(zgrid).to(u.Mpc).value)
    
    zgrid = jnp.array(zgrid)
    rs = jnp.asarray(rs)
    rs = rs.reshape(len(Om0grid), len(zgrid))
    
    @jit
    def E(z, Om0=Om0_fiducial):
        """Hubble parameter as function of redshift."""
        return jnp.sqrt(Om0 * (1 + z)**3 + (1.0 - Om0))
    
    @jit
    def r_of_z(z, H0, Om0=Om0_fiducial):
        """Comoving distance as function of redshift."""
        return interp2d(Om0, z, Om0grid, zgrid, rs) * (H0_fiducial / H0)
    
    @jit
    def dL_of_z(z, H0, Om0=Om0_fiducial):
        """Luminosity distance as function of redshift."""
        return (1 + z) * r_of_z(z, H0, Om0)
    
    @jit
    def z_of_dL(dL, H0, Om0=Om0_fiducial):
        """Redshift as function of luminosity distance."""
        return jnp.interp(dL, dL_of_z(zgrid, H0, Om0), zgrid)
    
    @jit
    def dV_of_z(z, H0, Om0=Om0_fiducial):
        """Comoving volume element as function of redshift."""
        return speed_of_light * r_of_z(z, H0, Om0)**2 / (H0 * E(z, Om0))
    
    @jit
    def ddL_of_z(z, dL, H0, Om0=Om0_fiducial):
        """Derivative of luminosity distance with respect to redshift."""
        return dL / (1 + z) + speed_of_light * (1 + z) / (H0 * E(z, Om0))
    
    return {
        'zgrid': zgrid,
        'Om0grid': Om0grid,
        'rs': rs,
        'E': E,
        'r_of_z': r_of_z,
        'dL_of_z': dL_of_z,
        'z_of_dL': z_of_dL,
        'dV_of_z': dV_of_z,
        'ddL_of_z': ddL_of_z,
        'H0_fiducial': H0_fiducial,
        'Om0_fiducial': Om0_fiducial
    }


def load_gw_samples(fn_gwsamples, N_gw_inf=None, N_samples_gw=None):
    """
    Load gravitational wave samples from HDF5 file.
    Loads separate galaxy and AGN arrays, concatenates them, shuffles, then selects N_gw_inf events.
    Flattens to 1D, converts to JAX arrays, and returns as dictionary.
    
    Parameters
    ----------
    fn_gwsamples : str
        Path to GW samples HDF5 file
    N_gw_inf : int, optional
        Number of GW events to use in inference. If None, uses all events.
        Must not exceed the total number of events available.
    N_samples_gw : int, optional
        Number of samples per event to use (default: None, uses all)
    
    Returns
    -------
    dict
        Dictionary containing:
        - ra: right ascension samples (JAX array, flattened to 1D)
        - dec: declination samples (JAX array, flattened to 1D)
        - dL: luminosity distance samples (JAX array, flattened to 1D)
        - p_pe: prior probability for each sample (JAX array, flattened to 1D)
        - N_samples_gw: number of samples per event
    """
    # Import here to avoid circular dependencies
    import generate_gwsamples
    
    # Load using the numpy-based function from generate_gwsamples
    # Returns separate arrays for galaxies and AGNs: (ra_gal, dec_gal, dL_gal, m1det_gal, m2det_gal, 
    #                                                   ra_agn, dec_agn, dL_agn, m1det_agn, m2det_agn)
    ra_gal, dec_gal, dL_gal, m1det_gal, m2det_gal, ra_agn, dec_agn, dL_agn, m1det_agn, m2det_agn = \
        generate_gwsamples.load_samples(fn_gwsamples, N_samples_gw=N_samples_gw)
    
    # Concatenate galaxies and AGNs
    ra = np.concatenate([ra_gal, ra_agn], axis=0)
    dec = np.concatenate([dec_gal, dec_agn], axis=0)
    dL = np.concatenate([dL_gal, dL_agn], axis=0)
    m1det = np.concatenate([m1det_gal, m1det_agn], axis=0)
    m2det = np.concatenate([m2det_gal, m2det_agn], axis=0)
    
    # Shuffle events to mix galaxies and AGNs
    N_gw_total = ra.shape[0]
    shuffle_indices = np.random.permutation(N_gw_total)
    ra = ra[shuffle_indices]
    dec = dec[shuffle_indices]
    dL = dL[shuffle_indices]
    m1det = m1det[shuffle_indices]
    m2det = m2det[shuffle_indices]
    
    # Select N_gw_inf events
    if N_gw_inf is not None:
        if N_gw_inf > N_gw_total:
            raise ValueError(f"N_gw_inf ({N_gw_inf}) exceeds total number of available events ({N_gw_total})")
        ra = ra[:N_gw_inf]
        dec = dec[:N_gw_inf]
        dL = dL[:N_gw_inf]
        m1det = m1det[:N_gw_inf]
        m2det = m2det[:N_gw_inf]
    
    # Flatten 2D arrays to 1D and convert to JAX arrays
    ra_flat = ra.flatten()
    dec_flat = dec.flatten()
    dL_flat = dL.flatten()
    p_pe = jnp.ones(len(ra_flat))
    
    # Get N_samples_gw and N_gw from the shape
    N_samples_gw_loaded = ra.shape[1] if len(ra) > 0 else 0
    N_gw_loaded = ra.shape[0] if len(ra) > 0 else 0
    print(f"N_samples_gw_loaded: {N_samples_gw_loaded}, N_gw_loaded: {N_gw_loaded}")
    
    return {
        'ra': jnp.array(ra_flat),
        'dec': jnp.array(dec_flat),
        'dL': jnp.array(dL_flat),
        'p_pe': p_pe,
        'N_samples_gw': N_samples_gw_loaded,
        'N_gw': N_gw_loaded
    }


def compute_pixel_indices(ra, dec, nside):
    """
    Compute Healpix pixel indices for given sky coordinates.
    
    Parameters
    ----------
    ra : array
        Right ascension in radians
    dec : array
        Declination in radians
    nside : int
        Healpix nside parameter
    
    Returns
    -------
    array
        Pixel indices
    """
    print(f"Computing pixel indices: nside={nside}, N_samples={len(ra)}")
    return hp.pixelfunc.ang2pix(nside, np.pi/2 - dec, ra)


def create_catalog_probability_functions(catalog_data):
    """
    Create catalog probability functions for galaxies and AGN.
    
    Parameters
    ----------
    catalog_data : dict
        Dictionary from load_catalog_data()
    
    Returns
    -------
    dict
        Dictionary containing:
        - logpcatalog_gals: log probability function for galaxies
        - logpcatalog_gals_vmap: vectorized version
        - logpcatalog_agns: log probability function for AGN
        - logpcatalog_agns_vmap: vectorized version
        - logPriorUniverse: combined prior function
    """
    print(f"Creating catalog probability functions: nside={catalog_data['nside']}")
    zgals = catalog_data['zgals']
    dzgals = catalog_data['dzgals']
    wgals = catalog_data['wgals']
    zagns = catalog_data['zagns']
    dzagns = catalog_data['dzagns']
    wagns = catalog_data['wagns']
    
    @jit
    def logpcatalog_gals(z, pix, Om0, gamma):
        """Log probability of redshift given galaxy catalog."""
        zs = zgals[pix]
        ddzs = dzgals[pix]
        valid_mask = jnp.isfinite(zs)  # Check for NaN (and inf) values used for padding
        ngals = jnp.sum(valid_mask)
        # Only normalize over valid entries
        wts = wgals[pix] * (1 + zs)**(gamma)
        wts_valid = jnp.where(valid_mask, wts, 0.0)
        wts_sum = jnp.sum(wts_valid)
        # Normalize only if there are valid entries
        wts_normalized = jnp.where(wts_sum > 0, wts_valid / wts_sum, 0.0)
        # Compute log probability - only include valid entries in the sum
        # Use a mask to exclude invalid entries from logsumexp
        log_wts = jnp.where(valid_mask & (wts_normalized > 0), 
                           jnp.log(wts_normalized) + norm.logpdf(z, zs, ddzs), 
                           -jnp.inf)
        log_prob = logsumexp(log_wts)
        # If no valid entries, return very low probability
        log_prob = jnp.where(ngals > 0, log_prob, -1e10)
        return log_prob, ngals
    
    logpcatalog_gals_vmap = jit(vmap(logpcatalog_gals, in_axes=(0, 0, None, None), out_axes=0))
    
    @jit
    def logpcatalog_agns(z, pix, Om0, gamma):
        """Log probability of redshift given AGN catalog."""
        zs = zagns[pix]
        ddzs = dzagns[pix]
        valid_mask = jnp.isfinite(zs)  # Check for NaN (and inf) values used for padding
        nagns = jnp.sum(valid_mask)
        # Only normalize over valid entries
        wts = wagns[pix] * (1 + zs)**(gamma)
        wts_valid = jnp.where(valid_mask, wts, 0.0)
        wts_sum = jnp.sum(wts_valid)
        # Normalize only if there are valid entries
        wts_normalized = jnp.where(wts_sum > 0, wts_valid / wts_sum, 0.0)
        # Compute log probability - only include valid entries in the sum
        # Use a mask to exclude invalid entries from logsumexp
        log_wts = jnp.where(valid_mask & (wts_normalized > 0), 
                           jnp.log(wts_normalized) + norm.logpdf(z, zs, ddzs), 
                           -jnp.inf)
        log_prob = logsumexp(log_wts)
        # If no valid entries, return very low probability
        log_prob = jnp.where(nagns > 0, log_prob, -1e10)
        return log_prob, nagns
    
    logpcatalog_agns_vmap = jit(vmap(logpcatalog_agns, in_axes=(0, 0, None, None), out_axes=0))
    
    def logPriorUniverse(z, pix, alpha_agn, Om0, gamma_agn, gamma_gal):
        """
        Combined log prior probability from galaxy and AGN catalogs.
        
        Parameters
        ----------
        z : array
            Redshifts
        pix : array
            Pixel indices
        alpha_agn : float
            Fraction of AGN hosts
        Om0 : float
            Matter density parameter
        gamma_agn : float
            AGN evolution parameter
        gamma_gal : float
            Galaxy evolution parameter
        
        Returns
        -------
        array
            Log prior probabilities
        """
        #print("agn vmapping")
        logpcat_agns, nagns = logpcatalog_agns_vmap(z, pix, Om0, gamma_agn)
        #logpcat_agns, nagns = logpcatalog_agns(z, pix, Om0, gamma_agn)
        #print("gals vmapping")
        logpcat_gals, ngals = logpcatalog_gals_vmap(z, pix, Om0, gamma_gal)
        #logpcat_gals, ngals = logpcatalog_gals(z, pix, Om0, gamma_gal)
        #print("log terms")
        # Handle edge cases: if no valid AGN/galaxies, set probability to very low value
        logpcat_agns = jnp.where(nagns > 0, logpcat_agns, -1e10)
        logpcat_gals = jnp.where(ngals > 0, logpcat_gals, -1e10)
        
        # Use safe log operations to avoid numerical issues
        # log1p(x) = log(1 + x)
        log_alpha_agn = jnp.where(alpha_agn > 1e-10, jnp.log(alpha_agn), -1e10)
        log_1malpha_agn = jnp.where(alpha_agn < 1.0 - 1e-10, jnp.log1p(-alpha_agn), -1e10)
        
        ### original
        ## alpha_agn * p_cat_agn + (1 - alpha_agn) * p_cat_gals
        # log_term1 = log_alpha_agn + logpcat_agns
        # log_term2 = log_1malpha_agn + logpcat_gals
        
        ### trying this cursor suggestion:
        # alpha_agn * N_agn * p_cat_agn + (1 - alpha_agn) * N_gal * p_cat_gals
        # Account for relative number of sources per pixel when combining probabilities
        # The catalog probabilities are normalized per pixel, but we need to weight them
        # by the relative probability of finding AGN vs galaxies in each pixel.
        # This prevents bias when pixels have different numbers of AGN vs galaxies.
        # Weight by relative number of sources (avoid division by zero)
        n_tot = nagns + ngals
        log_weight_agn = jnp.where(n_tot > 0, jnp.log(nagns + 1e-10) - jnp.log(n_tot + 1e-10), 0.0)
        log_weight_gal = jnp.where(n_tot > 0, jnp.log(ngals + 1e-10) - jnp.log(n_tot + 1e-10), 0.0)
        log_term1 = log_alpha_agn + logpcat_agns + log_weight_agn
        log_term2 = log_1malpha_agn + logpcat_gals + log_weight_gal
        
        #print("sum")
        log_prob = jnp.logaddexp(log_term1, log_term2)
        return log_prob
    
    return {
        'logpcatalog_gals': logpcatalog_gals,
        'logpcatalog_gals_vmap': logpcatalog_gals_vmap,
        'logpcatalog_agns': logpcatalog_agns,
        'logpcatalog_agns_vmap': logpcatalog_agns_vmap,
        'logPriorUniverse': logPriorUniverse
    }


def compute_darksiren_log_likelihood(
    gw_data, catalog_data, cosmo_funcs, prob_funcs,
    H0, alpha_agn, samples_ind, Om0=None, gamma_agn=0, gamma_gal=0
):
    """
    Compute dark siren log-likelihood.
    
    Parameters
    ----------
    gw_data : dict
        Dictionary from load_gw_samples()
    catalog_data : dict
        Dictionary from load_catalog_data()
    cosmo_funcs : dict
        Dictionary from setup_cosmology()
    prob_funcs : dict
        Dictionary from create_catalog_probability_functions()
    H0 : float
        Hubble constant
    alpha_agn : float
        Fraction of AGN hosts
    samples_ind : array
        Precomputed pixel indices for GW samples
    Om0 : float, optional
        Matter density parameter (default: Planck value)
    gamma_agn : float
        AGN evolution parameter (default: 0)
    gamma_gal : float
        Galaxy evolution parameter (default: 0)
    
    Returns
    -------
    float
        Log-likelihood value
    """
    s = time.time()
    #print(f"Computing dark siren log-likelihood: H0={H0}, alpha_agn={alpha_agn}, Om0={Om0}, gamma_agn={gamma_agn}, gamma_gal={gamma_gal}")
    if Om0 is None:
        # TODO for now using fiducial Om0, but should use Om0 from config
        # if we are not varying it
        Om0 = cosmo_funcs['Om0_fiducial']
    
    dL = gw_data['dL']
    p_pe = gw_data['p_pe']
    N_samples_gw = gw_data['N_samples_gw']
    N_gw = gw_data['N_gw']
    assert N_gw == len(dL) // N_samples_gw, "N_gw and N_samples_gw doe not match the loaded number of samples"

    z_of_dL = cosmo_funcs['z_of_dL']
    ddL_of_z = cosmo_funcs['ddL_of_z']
    logPriorUniverse = prob_funcs['logPriorUniverse']
    
    # Convert distances to redshifts
    #print(f"Computing redshifts: dL={dL}, H0={H0}, Om0={Om0}")
    z = z_of_dL(dL, H0, Om0)
    
    # Compute log weights
    #print(f"Computing log weights: z={z}, p_pe={p_pe}, N_gw={N_gw}, N_samples_gw={N_samples_gw}")
    log_weights = (
        -jnp.log(ddL_of_z(z, dL, H0, Om0)) 
        - jnp.log(p_pe) 
        + logPriorUniverse(z, samples_ind, alpha_agn, Om0, gamma_agn, gamma_gal)
    )
    e = time.time()
    #print(f"likelihood call time: {e - s} seconds")
    # Reshape and compute log-likelihood
    #print(f"Reshaping and computing log-likelihood: log_weights={log_weights}, N_gw={N_gw}, N_samples_gw={N_samples_gw}")
    log_weights = log_weights.reshape((N_gw, N_samples_gw))
    ll = jnp.sum(-jnp.log(N_samples_gw) + logsumexp(log_weights, axis=-1))
    
    return ll


def compute_likelihood_grid(
    gw_data, catalog_data, cosmo_funcs, prob_funcs,
    H0_grid, alpha_agn_grid, Om0=None, gamma_agn=0, gamma_gal=0,
    progress=True
):
    """
    Compute dark siren log-likelihood for a grid of parameters.
    
    Parameters
    ----------
    gw_data : dict
        Dictionary from load_gw_samples()
    catalog_data : dict
        Dictionary from load_catalog_data()
    cosmo_funcs : dict
        Dictionary from setup_cosmology()
    prob_funcs : dict
        Dictionary from create_catalog_probability_functions()
    H0_grid : array
        1D array of H0 values to evaluate
    alpha_agn_grid : array
        1D array of alpha_agn values to evaluate
    Om0 : float, optional
        Matter density parameter (default: Planck value)
    gamma_agn : float
        AGN evolution parameter (default: 0)
    gamma_gal : float
        Galaxy evolution parameter (default: 0)
    progress : bool
        Whether to show progress bar (default: True)
    
    Returns
    -------
    array
        2D array of log-likelihood values with shape (len(H0_grid), len(alpha_agn_grid))
        where log_likelihood[i, j] corresponds to H0_grid[i], alpha_agn_grid[j]
    """
    print(f"Computing likelihood grid: H0_grid shape={len(H0_grid)}, alpha_agn_grid shape={len(alpha_agn_grid)}, Om0={Om0}, gamma_agn={gamma_agn}, gamma_gal={gamma_gal}")
    
    # Precompute pixel indices once (they don't depend on H0, alpha_agn, Om0, or gammas)
    nside = catalog_data['nside']
    samples_ind = compute_pixel_indices(gw_data['ra'], gw_data['dec'], nside)
    
    if Om0 is None:
        Om0 = cosmo_funcs['Om0_fiducial']
    
    # Convert to JAX arrays if needed
    H0_grid = jnp.asarray(H0_grid)
    alpha_agn_grid = jnp.asarray(alpha_agn_grid)
    
    # Initialize output array
    log_likelihood_grid = jnp.zeros((len(H0_grid), len(alpha_agn_grid)))
    
    # Compute likelihood for each combination
    if progress:
        total = len(H0_grid) * len(alpha_agn_grid)
        pbar = tqdm(total=total, desc="Computing likelihood grid")
    
    for i, H0 in enumerate(H0_grid):
        for j, alpha_agn in enumerate(alpha_agn_grid):
            ll = compute_darksiren_log_likelihood(
                gw_data, catalog_data, cosmo_funcs, prob_funcs,
                float(H0), float(alpha_agn), samples_ind, Om0, gamma_agn, gamma_gal
            )
            log_likelihood_grid = log_likelihood_grid.at[i, j].set(float(ll))
            if progress:
                pbar.update(1)
    
    if progress:
        pbar.close()
    
    return np.array(log_likelihood_grid)


def save_likelihood_grid(
    fn_inf, log_likelihood_grid, H0_grid, alpha_agn_grid,
    fn_config=None, grid_params=None
):
    """
    Save likelihood grid and associated parameters to HDF5 file.
    
    Parameters
    ----------
    fn_inf : str
        Output HDF5 filename for inference results
    log_likelihood_grid : array
        2D array of log-likelihood values with shape (len(H0_grid), len(alpha_agn_grid))
    H0_grid : array
        1D array of H0 values used
    alpha_agn_grid : array
        1D array of alpha_agn values used
    fn_config : str, optional
        Path to inference config file
    grid_params : dict, optional
        Dictionary with grid computation parameters:
        - Om0: matter density parameter
        - gamma_agn: AGN evolution parameter
        - gamma_gal: galaxy evolution parameter
    """
    print(f"Saving likelihood grid to {fn_inf} (shape={log_likelihood_grid.shape})")
    os.makedirs(os.path.dirname(fn_inf) if os.path.dirname(fn_inf) else '.', exist_ok=True)
    
    with h5py.File(fn_inf, 'w') as f:
        # Save likelihood grid
        f.create_dataset('log_likelihood_grid', data=np.array(log_likelihood_grid))
        
        # Save parameter grids
        f.create_dataset('H0_grid', data=np.array(H0_grid))
        f.create_dataset('alpha_agn_grid', data=np.array(alpha_agn_grid))
        
        # Save grid parameters
        if grid_params is not None:
            grid_group = f.create_group('grid_params')
            for key, value in grid_params.items():
                if isinstance(value, str):
                    grid_group.attrs[key] = value
                elif isinstance(value, (int, float)):
                    grid_group.attrs[key] = value
                elif isinstance(value, bool):
                    grid_group.attrs[key] = int(value)
                elif value is None:
                    grid_group.attrs[key] = 'None'
                else:
                    grid_group.attrs[key] = str(value)
        
        # Save config file path
        if fn_config is not None:
            f.attrs['fn_config'] = fn_config
        
        # Save attributes
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['N_H0'] = len(H0_grid)
        f.attrs['N_alpha_agn'] = len(alpha_agn_grid)
        f.attrs['H0_min'] = float(np.min(H0_grid))
        f.attrs['H0_max'] = float(np.max(H0_grid))
        f.attrs['alpha_agn_min'] = float(np.min(alpha_agn_grid))
        f.attrs['alpha_agn_max'] = float(np.max(alpha_agn_grid))


def load_likelihood_grid(fn_inf):
    """
    Load likelihood grid and associated parameters from HDF5 file.
    
    Parameters
    ----------
    fn_inf : str
        Input HDF5 filename for inference results
    
    Returns
    -------
    dict
        Dictionary containing:
        - log_likelihood_grid: 2D array of log-likelihood values
        - H0_grid: 1D array of H0 values
        - alpha_agn_grid: 1D array of alpha_agn values
        - grid_params: dictionary with grid computation parameters
        - fn_config: path to inference config file
        - timestamp: timestamp string
    """
    print(f"Loading likelihood grid from {fn_inf}")
    results = {}
    
    with h5py.File(fn_inf, 'r') as f:
        # Load likelihood grid
        results['log_likelihood_grid'] = np.array(f['log_likelihood_grid'])
        
        # Load parameter grids
        results['H0_grid'] = np.array(f['H0_grid'])
        results['alpha_agn_grid'] = np.array(f['alpha_agn_grid'])
        
        # Load grid parameters
        if 'grid_params' in f:
            grid_params = {}
            for key in f['grid_params'].attrs:
                value = f['grid_params'].attrs[key]
                # Try to convert back to appropriate type
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                if value == 'None':
                    value = None
                elif isinstance(value, str):
                    # Try to parse as float/int if possible
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except (ValueError, TypeError):
                        pass
                grid_params[key] = value
            results['grid_params'] = grid_params
        
        # Load attributes
        results['fn_config'] = f.attrs.get('fn_config', None)
        results['timestamp'] = f.attrs.get('timestamp', None)
        results['N_H0'] = f.attrs.get('N_H0', f.attrs.get('n_H0', None))
        results['N_alpha_agn'] = f.attrs.get('N_alpha_agn', None)
        results['H0_min'] = f.attrs.get('H0_min', None)
        results['H0_max'] = f.attrs.get('H0_max', None)
        results['alpha_agn_min'] = f.attrs.get('alpha_agn_min', None)
        results['alpha_agn_max'] = f.attrs.get('alpha_agn_max', None)
    
    return results


def setup_mcmc_parameters(
    H0_bounds=(50, 100),
    alpha_agn_bounds=(0, 1),
    Om0_bounds=None,
    gamma_agn_bounds=(-5, 5),
    gamma_gal_bounds=(-5, 5)
):
    """
    Set up MCMC parameter bounds and labels.
    
    Parameters
    ----------
    H0_bounds : tuple
        (lower, upper) bounds for H0
    alpha_agn_bounds : tuple
        (lower, upper) bounds for alpha_agn
    Om0_bounds : tuple, optional
        (lower, upper) bounds for Om0 (default: from cosmology setup)
    gamma_agn_bounds : tuple
        (lower, upper) bounds for gamma_agn parameter
    gamma_gal_bounds : tuple
        (lower, upper) bounds for gamma_gal parameter
    
    Returns
    -------
    dict
        Dictionary containing:
        - lower_bound: list of lower bounds
        - upper_bound: list of upper bounds
        - labels: list of parameter labels
        - ndims: number of dimensions
    """
    print(f"Setting up MCMC parameters: H0_bounds={H0_bounds}, alpha_agn_bounds={alpha_agn_bounds}")
    lower_bound = [H0_bounds[0], alpha_agn_bounds[0]]
    upper_bound = [H0_bounds[1], alpha_agn_bounds[1]]
    
    labels = ['H0', 'alpha_agn']
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'labels': labels,
        'ndims': len(lower_bound)
    }


def create_mcmc_likelihood_function(
    gw_data, catalog_data, cosmo_funcs, prob_funcs,
    lower_bound, upper_bound, Om0=None, gamma_agn=0, gamma_gal=0
):
    """
    Create MCMC likelihood function with bounds checking.
    
    Parameters
    ----------
    gw_data : dict
        Dictionary from load_gw_samples()
    catalog_data : dict
        Dictionary from load_catalog_data()
    cosmo_funcs : dict
        Dictionary from setup_cosmology()
    prob_funcs : dict
        Dictionary from create_catalog_probability_functions()
    lower_bound : list
        Lower bounds for parameters
    upper_bound : list
        Upper bounds for parameters
    Om0 : float, optional
        Matter density parameter (default: Planck value)
    gamma_agn : float
        AGN evolution parameter (default: 0)
    gamma_gal : float
        Galaxy evolution parameter (default: 0)
    
    Returns
    -------
    function
        Likelihood function for MCMC
    """
    print(f"Creating MCMC likelihood function: lower_bound={lower_bound}, upper_bound={upper_bound}, Om0={Om0}, gamma_agn={gamma_agn}, gamma_gal={gamma_gal}")
    
    # Precompute pixel indices once (they don't depend on H0, alpha_agn, Om0, or gammas)
    nside = catalog_data['nside']
    samples_ind = compute_pixel_indices(gw_data['ra'], gw_data['dec'], nside)
    
    def likelihood_emcee(coord):
        """Likelihood function with bounds checking for emcee."""
        for i in range(len(coord)):
            if (coord[i] < lower_bound[i] or coord[i] > upper_bound[i]):
                return -np.inf
        H0, alpha_agn = coord
        ll = compute_darksiren_log_likelihood(
            gw_data, catalog_data, cosmo_funcs, prob_funcs,
            H0, alpha_agn, samples_ind, Om0, gamma_agn, gamma_gal
        )
        if np.isnan(ll):
            return -np.inf
        else:
            return float(ll)
    
    return likelihood_emcee


def solve_fagn_lambda(alpha_agn_obs, N_gal, N_agn):
    """
    Solve for f_agn and lambda_agn given observed AGN fraction.
    
    Parameters
    ----------
    alpha_agn_obs : float
        Observed fraction of AGN hosts
    N_gal : float
        Total number of galaxies
    N_agn : float
        Total number of AGN
    
    Returns
    -------
    tuple
        (f_agn, lambda_agn) solution
    """
    
    def loss(params):
        """Loss function for optimization."""
        f_agn, lambda_agn = params
        if not (0 <= f_agn <= 1 and 0 <= lambda_agn <= 1):
            return np.inf
        _, alpha_agn_model = utils.compute_gw_host_fractions(N_gal, N_agn, f_agn, lambda_agn)
        return (alpha_agn_model - alpha_agn_obs)**2
    
    # Initial guess: equal mixing and moderate AGN fraction
    x0 = [0.5, 0.5]
    bounds = [(0, 1), (0, 1)]
    
    result = minimize(loss, x0, bounds=bounds)
    if result.success:
        f_agn, lambda_agn = result.x
        return f_agn, lambda_agn
    else:
        raise RuntimeError("Optimization failed")


def run_mcmc_sampling(
    likelihood_func, lower_bound, upper_bound, N_walkers=64, N_steps=1000,
    seed=None
):
    """
    Run MCMC sampling using emcee.
    
    Parameters
    ----------
    likelihood_func : function
        Likelihood function for MCMC
    lower_bound : list
        Lower bounds for parameters
    upper_bound : list
        Upper bounds for parameters
    N_walkers : int
        Number of walkers (default: 64)
    N_steps : int
        Number of MCMC steps (default: 1000)
    seed : int, optional
        Random seed for initialization
    
    Returns
    -------
    emcee.EnsembleSampler
        Sampler object with chain
    """
    print(f"Running MCMC sampling: N_walkers={N_walkers}, N_steps={N_steps}, seed={seed}, ndims={len(lower_bound)}")
    if emcee is None:
        raise ImportError("emcee is required for MCMC sampling. Install with: pip install emcee")
    
    ndims = len(lower_bound)
    
    if seed is not None:
        np.random.seed(seed)
    
    p0 = np.random.uniform(lower_bound, upper_bound, size=(N_walkers, ndims))
    
    sampler = emcee.EnsembleSampler(
        N_walkers, ndims, likelihood_func,
        moves=[
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ]
    )
    
    sampler.run_mcmc(p0, N_steps, progress=True)
    
    return sampler


def get_posterior_samples(sampler, burnin_frac=0.5, N_samples=None):
    """
    Extract posterior samples from MCMC chain.
    
    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Sampler object with chain
    burnin_frac : float
        Fraction of chain to discard as burn-in (default: 0.5)
    N_samples : int, optional
        Number of samples to return (default: all post-burnin)
    
    Returns
    -------
    array
        Posterior samples
    """
    print(f"Extracting posterior samples: burnin_frac={burnin_frac}, N_samples={N_samples}")
    shape = sampler.flatchain.shape[0]
    burnin_idx = int(shape * burnin_frac)
    samples = sampler.flatchain[burnin_idx:, :]
    
    if N_samples is not None and N_samples < len(samples):
        choose = np.random.randint(0, len(samples), N_samples)
        samples = samples[choose]
    
    return samples


if __name__ == "__main__":
    config_inference, config_data, fn_config, overwrite = parse_args()
    main(config_inference, config_data, fn_config, overwrite=overwrite)        
