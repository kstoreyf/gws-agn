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
if '--undefok' not in ' '.join(sys.argv):
    sys.argv.insert(1, '--undefok=coordination_agent_recoverable')

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

import utils

try:
    import emcee
except ImportError:
    emcee = None



def main(
    f_agn=0.5, lambda_agn=0.5, N_gw=1000, gw_seed=1042, nside=256,
    n_walkers=16, n_steps=1000, burnin_frac=0.2,
    Om0=None, gamma_agn=0, gamma_gal=0,
    H0_bounds=(50, 100), f_bounds=(0, 1),
    mode='mcmc', n_H0=20, n_f=20,
    seed=None
):
    """
    Main function to run the inference pipeline.
    
    Parameters
    ----------
    f_agn : float
        Fraction of AGN hosts (default: 0.5)
    lambda_agn : float
        AGN mixing parameter (default: 0.5)
    N_gw : int
        Number of GW events (default: 1000)
    gw_seed : int
        Random seed for GW samples (default: 1042)
    nside : int
        Healpix nside parameter (default: 64)
    n_walkers : int
        Number of MCMC walkers (default: 16)
    n_steps : int
        Number of MCMC steps (default: 1000)
    burnin_frac : float
        Burn-in fraction (default: 0.2)
    Om0 : float, optional
        Matter density parameter (default: None, uses Planck value)
    gamma_agn : float
        AGN evolution parameter (default: 0)
    gamma_gal : float
        Galaxy evolution parameter (default: 0)
    H0_bounds : tuple
        Bounds for H0 parameter (default: (20, 120))
    f_bounds : tuple
        Bounds for f parameter (default: (0, 1))
    mode : str
        Mode to run: 'mcmc' or 'likelihood_grid' (default: 'mcmc')
    n_H0 : int
        Number of H0 grid points for likelihood_grid mode (default: 50)
    n_f : int
        Number of f grid points for likelihood_grid mode (default: 50)
    seed : int, optional
        Random seed for MCMC initialization (default: None)
    
    Returns
    -------
    dict
        Dictionary containing results depending on mode:
        - For 'mcmc': posterior_samples, sampler, config, metadata, mcmc_params
        - For 'likelihood_grid': log_likelihood_grid, H0_grid, f_grid, config, metadata
    """
    #mode = 'likelihood_grid'
    mode = 'mcmc'

    ratioNgalNagn = 1
    bias_gal = 1.0
    bias_agn = 1.0
    f_agn = 0.5
    lambda_agn = 0.0
    N_gw = 1000
    seed = 42
    gw_seed = 1042

    print(f"Running main inference pipeline: mode={mode}, f_agn={f_agn}, lambda_agn={lambda_agn}, N_gw={N_gw}, nside={nside}, n_walkers={n_walkers}, n_steps={n_steps}")

    tag_mock_extra = f'_bgal{bias_gal}_bagn{bias_agn}'
    tag_mock = f'_seed{seed}_ratioNgalNagn{ratioNgalNagn}{tag_mock_extra}'
    dir_mock = f'../data/mocks_glass/mock{tag_mock}'
    galaxy_file = os.path.join(dir_mock, f'lognormal_pixelated_nside_{nside}_galaxies.h5')
    agn_file = os.path.join(dir_mock, f'lognormal_pixelated_nside_{nside}_agn.h5')
    # NOTE rn named pos_only; think about when we move to more complex inf
    #gw_file = os.path.join(dir_mock, f'gwsamples_fagn{f_agn}_lambdaagn{lambda_agn}_N{N_gw}_seed{gw_seed}_pos_only.h5')
    gw_file = os.path.join(dir_mock, f'gwsamples_fagn{f_agn}_lambdaagn{lambda_agn}_N{N_gw}_seed{gw_seed}.h5')

    tag_inf = f'_fagn{f_agn}_lambdaagn{lambda_agn}_N{N_gw}_seed{gw_seed}_nside{nside}_norm'
    output_file = f'../results/inference/inference_results{tag_inf}.h5'

    #Check JAX GPU status
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"GPU devices: {[d for d in jax.devices() if d.device_kind == 'gpu']}")
    #print(f"Test array device: {jnp.array([1.0]).device()}")

    # Load catalog data
    catalog_data = load_catalog_data(galaxy_file, agn_file, nside=nside)
    
    # Setup cosmology
    cosmo_funcs = setup_cosmology()
    
    # Create catalog probability functions
    prob_funcs = create_catalog_probability_functions(catalog_data)

    # Load GW samples
    gw_data = load_gw_samples(gw_file)

    # Set up MCMC parameters
    mcmc_params = setup_mcmc_parameters(H0_bounds=H0_bounds, f_bounds=f_bounds)

    # Run inference based on mode
    if mode == 'likelihood_grid':
        results = run_likelihood_grid(
            gw_data, catalog_data, cosmo_funcs, prob_funcs, mcmc_params,
            galaxy_file, agn_file, gw_file, nside,
            Om0=Om0, gamma_agn=gamma_agn, gamma_gal=gamma_gal,
            n_H0=n_H0, n_f=n_f,
            output_file=output_file,
            f_agn=f_agn, lambda_agn=lambda_agn, N_gw=N_gw, gw_seed=gw_seed,
            seed=seed, ratioNgalNagn=ratioNgalNagn, bias_gal=bias_gal, bias_agn=bias_agn
        )
    elif mode == 'mcmc':
        results = run_inference_mcmc(
            gw_data, catalog_data, cosmo_funcs, prob_funcs, mcmc_params,
            galaxy_file, agn_file, gw_file, nside,
            n_walkers=n_walkers, n_steps=n_steps, burnin_frac=burnin_frac,
            Om0=Om0, gamma_agn=gamma_agn, gamma_gal=gamma_gal,
            H0_bounds=H0_bounds, f_bounds=f_bounds,
            seed=seed, output_file=output_file,
            f_agn=f_agn, lambda_agn=lambda_agn, N_gw=N_gw, gw_seed=gw_seed,
            ratioNgalNagn=ratioNgalNagn, bias_gal=bias_gal, bias_agn=bias_agn
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'mcmc' or 'likelihood_grid'")
    
    return results


def run_likelihood_grid(
    gw_data, catalog_data, cosmo_funcs, prob_funcs, mcmc_params,
    galaxy_file, agn_file, gw_file, nside,
    Om0=None, gamma_agn=0, gamma_gal=0,
    n_H0=50, n_f=50,
    output_file=None,
    f_agn=None, lambda_agn=None, N_gw=None, gw_seed=None,
    seed=None, ratioNgalNagn=None, bias_gal=None, bias_agn=None
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
    galaxy_file : str
        Path to galaxy catalog file
    agn_file : str
        Path to AGN catalog file
    gw_file : str
        Path to GW samples file
    nside : int
        Healpix nside parameter
    Om0 : float, optional
        Matter density parameter (default: None, uses Planck value)
    gamma_agn : float
        AGN evolution parameter (default: 0)
    gamma_gal : float
        Galaxy evolution parameter (default: 0)
    n_H0 : int
        Number of H0 grid points (default: 50)
    n_f : int
        Number of f grid points (default: 50)
    output_file : str, optional
        Path to save likelihood grid (default: None)
    
    Returns
    -------
    dict
        Dictionary containing:
        - log_likelihood_grid: 2D array of log-likelihood values
        - H0_grid: 1D array of H0 values
        - f_grid: 1D array of f values
        - config: configuration dictionary
        - metadata: metadata dictionary
    """
    print(f"Running likelihood grid computation: n_H0={n_H0}, n_f={n_f}")
    
    # Create parameter grids from bounds
    H0_grid = np.linspace(mcmc_params['lower_bound'][0], mcmc_params['upper_bound'][0], n_H0)
    f_grid = np.linspace(mcmc_params['lower_bound'][1], mcmc_params['upper_bound'][1], n_f)
    
    # Compute likelihood grid
    log_likelihood_grid = compute_likelihood_grid(
        gw_data, catalog_data, cosmo_funcs, prob_funcs,
        H0_grid, f_grid,
        Om0=Om0, gamma_agn=gamma_agn, gamma_gal=gamma_gal,
        progress=True
    )
    
    # Create configuration and metadata
    config = create_inference_config(
        galaxy_file, agn_file, gw_file, nside,
        gw_data['nEvents'], gw_data['nsamp'],
        n_walkers=None, n_steps=None, burnin_frac=None,
        Om0=Om0, gamma_agn=gamma_agn, gamma_gal=gamma_gal,
        H0_bounds=(mcmc_params['lower_bound'][0], mcmc_params['upper_bound'][0]),
        f_bounds=(mcmc_params['lower_bound'][1], mcmc_params['upper_bound'][1]),
        f_agn=f_agn, lambda_agn=lambda_agn, N_gw=N_gw, gw_seed=gw_seed,
        seed=seed, ratioNgalNagn=ratioNgalNagn, bias_gal=bias_gal, bias_agn=bias_agn
    )
    config['n_H0'] = n_H0
    config['n_f'] = n_f
    config['mode'] = 'likelihood_grid'
    
    metadata = create_inference_metadata(catalog_data, cosmo_funcs, gw_data)
    
    # Save results if output file specified
    if output_file is not None:
        # Modify output filename for grid mode
        grid_output_file = output_file.replace('inference_results', 'likelihood_grid')
        save_likelihood_grid(
            grid_output_file,
            log_likelihood_grid, H0_grid, f_grid,
            config=config,
            metadata=metadata,
            grid_params={'Om0': Om0, 'gamma_agn': gamma_agn, 'gamma_gal': gamma_gal}
        )
        print(f"Likelihood grid saved to {grid_output_file}")
    
    return {
        'log_likelihood_grid': log_likelihood_grid,
        'H0_grid': H0_grid,
        'f_grid': f_grid,
        'config': config,
        'metadata': metadata,
        'mcmc_params': mcmc_params
    }


def run_inference_mcmc(
    gw_data, catalog_data, cosmo_funcs, prob_funcs, mcmc_params,
    galaxy_file, agn_file, gw_file, nside,
    n_walkers=16, n_steps=1000, burnin_frac=0.2,
    Om0=None, gamma_agn=0, gamma_gal=0,
    H0_bounds=(20, 120), f_bounds=(0, 1),
    seed=None, output_file=None,
    f_agn=None, lambda_agn=None, N_gw=None, gw_seed=None,
    ratioNgalNagn=None, bias_gal=None, bias_agn=None
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
    galaxy_file : str
        Path to galaxy catalog file
    agn_file : str
        Path to AGN catalog file
    gw_file : str
        Path to GW samples file
    nside : int
        Healpix nside parameter
    n_walkers : int
        Number of MCMC walkers (default: 16)
    n_steps : int
        Number of MCMC steps (default: 1000)
    burnin_frac : float
        Burn-in fraction (default: 0.2)
    Om0 : float, optional
        Matter density parameter (default: None, uses Planck value)
    gamma_agn : float
        AGN evolution parameter (default: 0)
    gamma_gal : float
        Galaxy evolution parameter (default: 0)
    H0_bounds : tuple
        Bounds for H0 parameter (default: (20, 120))
    f_bounds : tuple
        Bounds for f parameter (default: (0, 1))
    seed : int, optional
        Random seed for MCMC initialization (default: None)
    output_file : str, optional
        Path to save inference results (default: None)
    
    Returns
    -------
    dict
        Dictionary containing:
        - posterior_samples: posterior samples array
        - sampler: emcee sampler object
        - config: configuration dictionary
        - metadata: metadata dictionary
        - mcmc_params: MCMC parameter dictionary
    """
    print(f"Running MCMC inference: n_walkers={n_walkers}, n_steps={n_steps}, burnin_frac={burnin_frac}")
    
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
        n_walkers=n_walkers,
        n_steps=n_steps,
        seed=seed
    )

    # Get posterior samples
    posterior_samples = get_posterior_samples(sampler, burnin_frac=burnin_frac)
    
    # Create configuration and metadata
    config = create_inference_config(
        galaxy_file, agn_file, gw_file, nside,
        gw_data['nEvents'], gw_data['nsamp'],
        n_walkers, n_steps, burnin_frac,
        Om0=Om0, gamma_agn=gamma_agn, gamma_gal=gamma_gal,
        H0_bounds=H0_bounds, f_bounds=f_bounds,
        f_agn=f_agn, lambda_agn=lambda_agn, N_gw=N_gw, gw_seed=gw_seed,
        seed=seed, ratioNgalNagn=ratioNgalNagn, bias_gal=bias_gal, bias_agn=bias_agn
    )
    config['mode'] = 'mcmc'
    
    metadata = create_inference_metadata(catalog_data, cosmo_funcs, gw_data)
    
    # Save results if output file specified
    if output_file is not None:
        save_inference_results(
            output_file,
            posterior_samples,
            sampler=sampler,
            mcmc_params=mcmc_params,
            config=config,
            metadata=metadata
        )
        print(f"Inference results saved to {output_file}")
    
    return {
        'posterior_samples': posterior_samples,
        'sampler': sampler,
        'config': config,
        'metadata': metadata,
        'mcmc_params': mcmc_params
    }


def save_inference_results(
    filename, posterior_samples, sampler=None, mcmc_params=None,
    config=None, metadata=None
):
    """
    Save inference results to HDF5 file.
    
    Parameters
    ----------
    filename : str
        Output HDF5 filename
    posterior_samples : array
        Posterior samples array (n_samples, n_params)
    sampler : emcee.EnsembleSampler, optional
        Full sampler object (saves full chain if provided)
    mcmc_params : dict, optional
        MCMC parameter dictionary from setup_mcmc_parameters()
    config : dict, optional
        Configuration dictionary with:
        - galaxy_file: path to galaxy catalog
        - agn_file: path to AGN catalog
        - gw_file: path to GW samples
        - nside: healpix nside
        - nEvents: number of GW events
        - nsamp: number of samples per event
        - n_walkers: number of MCMC walkers
        - n_steps: number of MCMC steps
        - burnin_frac: burn-in fraction
        - Om0: matter density parameter
        - gamma_agn: AGN evolution parameter
        - gamma_gal: galaxy evolution parameter
    metadata : dict, optional
        Additional metadata to save
    """
    print(f"Saving inference results to {filename} (n_samples={len(posterior_samples)})")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with h5py.File(filename, 'w') as f:
        # Save posterior samples
        f.create_dataset('posterior_samples', data=np.array(posterior_samples))
        
        # Save full chain if sampler provided
        if sampler is not None:
            f.create_dataset('mcmc_chain', data=np.array(sampler.chain))
            f.create_dataset('mcmc_log_prob', data=np.array(sampler.lnprobability))
            f.attrs['n_walkers'] = sampler.nwalkers
            f.attrs['n_steps'] = sampler.iteration
        
        # Save MCMC parameters
        if mcmc_params is not None:
            f.create_dataset('lower_bound', data=np.array(mcmc_params['lower_bound']))
            f.create_dataset('upper_bound', data=np.array(mcmc_params['upper_bound']))
            f.attrs['ndims'] = mcmc_params['ndims']
            # Save labels as JSON string (since HDF5 doesn't support lists of strings well)
            if 'labels' in mcmc_params:
                f.attrs['labels'] = json.dumps(mcmc_params['labels'])
        
        # Save configuration
        if config is not None:
            config_group = f.create_group('config')
            for key, value in config.items():
                if isinstance(value, str):
                    config_group.attrs[key] = value
                elif isinstance(value, (int, float)):
                    config_group.attrs[key] = value
                elif isinstance(value, bool):
                    config_group.attrs[key] = int(value)
                elif isinstance(value, (tuple, list)):
                    # Convert tuples/lists to JSON string
                    config_group.attrs[key] = json.dumps(value)
                elif value is None:
                    config_group.attrs[key] = 'None'
                else:
                    # Convert other types to string
                    config_group.attrs[key] = str(value)
        
        # Save metadata
        if metadata is not None:
            metadata_group = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, str):
                    metadata_group.attrs[key] = value
                elif isinstance(value, (int, float)):
                    metadata_group.attrs[key] = value
                elif isinstance(value, bool):
                    metadata_group.attrs[key] = int(value)
                elif value is None:
                    metadata_group.attrs[key] = 'None'
                else:
                    metadata_group.attrs[key] = str(value)
        
        # Save timestamp
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['n_samples'] = len(posterior_samples)
        f.attrs['n_params'] = posterior_samples.shape[1] if len(posterior_samples.shape) > 1 else 1


def load_inference_results(filename):
    """
    Load inference results from HDF5 file.
    
    Parameters
    ----------
    filename : str
        Input HDF5 filename
    
    Returns
    -------
    dict
        Dictionary containing:
        - posterior_samples: array of posterior samples
        - mcmc_chain: full MCMC chain (if available)
        - mcmc_log_prob: log probabilities (if available)
        - mcmc_params: MCMC parameter dictionary
        - config: configuration dictionary
        - metadata: metadata dictionary
        - timestamp: timestamp string
        - n_samples: number of samples
        - n_params: number of parameters
    """
    print(f"Loading inference results from {filename}")
    results = {}
    
    with h5py.File(filename, 'r') as f:
        # Load posterior samples
        results['posterior_samples'] = np.array(f['posterior_samples'])
        
        # Load full chain if available
        if 'mcmc_chain' in f:
            results['mcmc_chain'] = np.array(f['mcmc_chain'])
            results['mcmc_log_prob'] = np.array(f['mcmc_log_prob'])
            results['n_walkers'] = f.attrs.get('n_walkers', None)
            results['n_steps'] = f.attrs.get('n_steps', None)
        
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
        
        # Load configuration
        if 'config' in f:
            config = {}
            for key in f['config'].attrs:
                value = f['config'].attrs[key]
                # Try to convert back to appropriate type
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                if value == 'None':
                    value = None
                # Try to parse as JSON (for tuples/lists)
                elif isinstance(value, str) and (value.startswith('[') or value.startswith('(')):
                    try:
                        value = json.loads(value)
                        # Convert lists to tuples if they were originally tuples
                        # (we can't distinguish, so keep as lists)
                    except (json.JSONDecodeError, ValueError):
                        pass
                config[key] = value
            results['config'] = config
        
        # Load metadata
        if 'metadata' in f:
            metadata = {}
            for key in f['metadata'].attrs:
                value = f['metadata'].attrs[key]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                if value == 'None':
                    value = None
                metadata[key] = value
            results['metadata'] = metadata
        
        # Load attributes
        results['timestamp'] = f.attrs.get('timestamp', None)
        results['n_samples'] = f.attrs.get('n_samples', None)
        results['n_params'] = f.attrs.get('n_params', None)
    
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
    
    if 'n_samples' in results and results['n_samples']:
        print(f"Number of samples: {results['n_samples']}")
    
    if 'n_params' in results and results['n_params']:
        print(f"Number of parameters: {results['n_params']}")
    
    if 'mcmc_params' in results:
        print("\nMCMC Parameters:")
        mcmc = results['mcmc_params']
        if 'labels' in mcmc:
            for i, label in enumerate(mcmc['labels']):
                lower = mcmc['lower_bound'][i]
                upper = mcmc['upper_bound'][i]
                print(f"  {label}: [{lower}, {upper}]")
    
    if 'config' in results:
        print("\nConfiguration:")
        config = results['config']
        for key in ['galaxy_file', 'agn_file', 'gw_file', 'nside', 
                   'nEvents', 'nsamp', 'n_walkers', 'n_steps']:
            if key in config:
                print(f"  {key}: {config[key]}")
    
    if 'metadata' in results:
        print("\nMetadata:")
        metadata = results['metadata']
        for key in ['H0Planck', 'Om0Planck', 'N_gal', 'N_agn']:
            if key in metadata:
                print(f"  {key}: {metadata[key]}")
    
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


def create_inference_config(
    galaxy_file, agn_file, gw_file, nside, nEvents, nsamp,
    n_walkers, n_steps, burnin_frac=0.5,
    Om0=None, gamma_agn=0, gamma_gal=0,
    H0_bounds=(20, 120), f_bounds=(0, 1),
    f_agn=None, lambda_agn=None, N_gw=None, gw_seed=None,
    seed=None, ratioNgalNagn=None, bias_gal=None, bias_agn=None
):
    """
    Create configuration dictionary for inference run.
    
    Parameters
    ----------
    galaxy_file : str
        Path to galaxy catalog file
    agn_file : str
        Path to AGN catalog file
    gw_file : str
        Path to GW samples file
    nside : int
        Healpix nside parameter
    nEvents : int
        Number of GW events
    nsamp : int
        Number of samples per event
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    burnin_frac : float
        Burn-in fraction (default: 0.5)
    Om0 : float, optional
        Matter density parameter
    gamma_agn : float
        AGN evolution parameter (default: 0)
    gamma_gal : float
        Galaxy evolution parameter (default: 0)
    H0_bounds : tuple
        Bounds for H0 parameter
    f_bounds : tuple
        Bounds for f parameter
    f_agn : float, optional
        Fraction of AGN hosts
    lambda_agn : float, optional
        AGN mixing parameter
    N_gw : int, optional
        Number of GW events
    gw_seed : int, optional
        Random seed for GW samples
    seed : int, optional
        Random seed for MCMC initialization
    ratioNgalNagn : float, optional
        Ratio of galaxies to AGN
    bias_gal : float, optional
        Galaxy bias parameter
    bias_agn : float, optional
        AGN bias parameter
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    print(f"Creating inference config: nEvents={nEvents}, nsamp={nsamp}, n_walkers={n_walkers}, n_steps={n_steps}, Om0={Om0}")
    config = {
        'galaxy_file': galaxy_file,
        'agn_file': agn_file,
        'gw_file': gw_file,
        'nside': nside,
        'nEvents': nEvents,
        'nsamp': nsamp,
        'n_walkers': n_walkers,
        'n_steps': n_steps,
        'burnin_frac': burnin_frac,
        'gamma_agn': gamma_agn,
        'gamma_gal': gamma_gal,
        'H0_bounds': H0_bounds,
        'f_bounds': f_bounds
    }
    
    if Om0 is not None:
        config['Om0'] = Om0
    
    # Add top-level parameters if provided
    if f_agn is not None:
        config['f_agn'] = f_agn
    if lambda_agn is not None:
        config['lambda_agn'] = lambda_agn
    if N_gw is not None:
        config['N_gw'] = N_gw
    if gw_seed is not None:
        config['gw_seed'] = gw_seed
    if seed is not None:
        config['seed'] = seed
    if ratioNgalNagn is not None:
        config['ratioNgalNagn'] = ratioNgalNagn
    if bias_gal is not None:
        config['bias_gal'] = bias_gal
    if bias_agn is not None:
        config['bias_agn'] = bias_agn
    
    return config


def create_inference_metadata(
    catalog_data, cosmo_funcs, gw_data
):
    """
    Create metadata dictionary from data objects.
    
    Parameters
    ----------
    catalog_data : dict
        Dictionary from load_catalog_data()
    cosmo_funcs : dict
        Dictionary from setup_cosmology()
    gw_data : dict
        Dictionary from load_gw_samples()
    
    Returns
    -------
    dict
        Metadata dictionary
    """
    print(f"Creating inference metadata: nEvents={gw_data['nEvents']}, nsamp={gw_data['nsamp']}")
    metadata = {
        'H0Planck': float(cosmo_funcs.get('H0Planck', None)),
        'Om0Planck': float(cosmo_funcs.get('Om0Planck', None)),
        'N_gal': float(catalog_data['ngals'].sum()),
        'N_agn': float(catalog_data['nagns'].sum()),
        'nside': int(catalog_data['nside']),
        'nEvents': int(gw_data['nEvents']),
        'nsamp': int(gw_data['nsamp'])
    }
    
    return metadata


def load_catalog_data(galaxy_file, agn_file, nside=256):
    """
    Load galaxy and AGN catalog data from HDF5 files.
    
    Parameters
    ----------
    galaxy_file : str
        Path to galaxy catalog HDF5 file
    agn_file : str
        Path to AGN catalog HDF5 file
    nside : int
        Healpix nside parameter (default: 256)
    
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
    print(f"Loading catalog data: galaxy_file={galaxy_file}, agn_file={agn_file}, nside={nside}")
    with h5py.File(galaxy_file, 'r') as f:
        # Use 'z' and 'n_in_pixel' dataset names (as saved by pixelize_catalogs.py)
        zgals = jnp.asarray(f['z'])
        dzgals = 0.0001 * (1 + zgals)
        wgals = jnp.ones(zgals.shape)
        ngals = jnp.asarray(f['n_in_pixel'])
    
    with h5py.File(agn_file, 'r') as f:
        # Use 'z' and 'n_in_pixel' dataset names (as saved by pixelize_catalogs.py)
        zagns = jnp.asarray(f['z'])
        dzagns = 0.0001 * (1 + zagns)
        wagns = jnp.ones(zagns.shape)
        nagns = jnp.asarray(f['n_in_pixel'])
    
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
    H0Planck = Planck15.H0.value
    Om0Planck = Planck15.Om0
    speed_of_light = constants.c.to('km/s').value
    
    # Create redshift grid
    zgrid_1 = np.expm1(np.linspace(np.log(1), np.log(zMax_1 + 1), 5000))
    zgrid_2 = np.expm1(np.linspace(np.log(zMax_1 + 1), np.log(zMax_2 + 1), 1000))
    zgrid = np.concatenate([zgrid_1, zgrid_2])
    
    # Create Om0 grid and compute comoving distances
    Om0grid = jnp.linspace(Om0Planck - Om0_range, Om0Planck + Om0_range, n_Om0)
    rs = []
    for Om0 in tqdm(Om0grid):
        cosmo = FlatLambdaCDM(H0=H0Planck, Om0=Om0)
        rs.append(cosmo.comoving_distance(zgrid).to(u.Mpc).value)
    
    zgrid = jnp.array(zgrid)
    rs = jnp.asarray(rs)
    rs = rs.reshape(len(Om0grid), len(zgrid))
    
    @jit
    def E(z, Om0=Om0Planck):
        """Hubble parameter as function of redshift."""
        return jnp.sqrt(Om0 * (1 + z)**3 + (1.0 - Om0))
    
    @jit
    def r_of_z(z, H0, Om0=Om0Planck):
        """Comoving distance as function of redshift."""
        return interp2d(Om0, z, Om0grid, zgrid, rs) * (H0Planck / H0)
    
    @jit
    def dL_of_z(z, H0, Om0=Om0Planck):
        """Luminosity distance as function of redshift."""
        return (1 + z) * r_of_z(z, H0, Om0)
    
    @jit
    def z_of_dL(dL, H0, Om0=Om0Planck):
        """Redshift as function of luminosity distance."""
        return jnp.interp(dL, dL_of_z(zgrid, H0, Om0), zgrid)
    
    @jit
    def dV_of_z(z, H0, Om0=Om0Planck):
        """Comoving volume element as function of redshift."""
        return speed_of_light * r_of_z(z, H0, Om0)**2 / (H0 * E(z, Om0))
    
    @jit
    def ddL_of_z(z, dL, H0, Om0=Om0Planck):
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
        'H0Planck': H0Planck,
        'Om0Planck': Om0Planck
    }


def load_gw_samples(filename, nEvents=1000, nsamp=None):
    """
    Load gravitational wave samples from HDF5 file.
    
    Parameters
    ----------
    filename : str
        Path to GW samples HDF5 file
    nEvents : int, optional
        Number of events to use (default: all)
    nsamp : int, optional
        Number of samples per event to use (default: all)
    
    Returns
    -------
    dict
        Dictionary containing:
        - ra: right ascension samples
        - dec: declination samples
        - dL: luminosity distance samples
        - p_pe: prior probability for each sample
        - nEvents: number of events
        - nsamp: number of samples per event
    """
    print(f"Loading GW samples from {filename} (nEvents={nEvents}, nsamp={nsamp})")
    with h5py.File(filename, 'r') as inp:
        nsamp_ = inp.attrs['nsamp']
        nEvents_ = inp.attrs['nobs']
        # Read as 2D arrays with shape (nEvents_, nsamp_)
        ra = jnp.array(inp['ra'])  # Shape: (nEvents_, nsamp_)
        dec = jnp.array(inp['dec'])  # Shape: (nEvents_, nsamp_)
        dL = jnp.array((jnp.array(inp['dL']) * u.Mpc).value)  # Shape: (nEvents_, nsamp_)
    
    if nsamp is None:
        nsamp = nsamp_
    if nEvents is None:
        nEvents = nEvents_
    
    # Slice to requested number of events and samples, then flatten to 1D
    ra = ra[0:nEvents, 0:nsamp].flatten()
    dec = dec[0:nEvents, 0:nsamp].flatten()
    dL = dL[0:nEvents, 0:nsamp].flatten()
    
    p_pe = jnp.ones(len(dL))
    
    return {
        'ra': ra,
        'dec': dec,
        'dL': dL,
        'p_pe': p_pe,
        'nEvents': nEvents,
        'nsamp': nsamp
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
    print(f"Computing pixel indices: nside={nside}, n_samples={len(ra)}")
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
    
    def logPriorUniverse(z, pix, f, Om0, gamma_agn, gamma_gal):
        """
        Combined log prior probability from galaxy and AGN catalogs.
        
        Parameters
        ----------
        z : array
            Redshifts
        pix : array
            Pixel indices
        f : float
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
        log_f = jnp.where(f > 1e-10, jnp.log(f), -1e10)
        log_1mf = jnp.where(f < 1.0 - 1e-10, jnp.log1p(-f), -1e10)
        
        ### original
        # log_term1 = log_f + logpcat_agns
        # log_term2 = log_1mf + logpcat_gals
        
        ### trying this cursor suggestion:
        # Account for relative number of sources per pixel when combining probabilities
        # The catalog probabilities are normalized per pixel, but we need to weight them
        # by the relative probability of finding AGN vs galaxies in each pixel.
        # This prevents bias when pixels have different numbers of AGN vs galaxies.
        n_tot = nagns + ngals
        # Weight by relative number of sources (avoid division by zero)
        log_weight_agn = jnp.where(n_tot > 0, jnp.log(nagns + 1e-10) - jnp.log(n_tot + 1e-10), 0.0)
        log_weight_gal = jnp.where(n_tot > 0, jnp.log(ngals + 1e-10) - jnp.log(n_tot + 1e-10), 0.0)
        
        log_term1 = log_f + logpcat_agns + log_weight_agn
        log_term2 = log_1mf + logpcat_gals + log_weight_gal
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
    H0, f, samples_ind, Om0=None, gamma_agn=0, gamma_gal=0
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
    f : float
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
    #print(f"Computing dark siren log-likelihood: H0={H0}, f={f}, Om0={Om0}, gamma_agn={gamma_agn}, gamma_gal={gamma_gal}")
    if Om0 is None:
        Om0 = cosmo_funcs['Om0Planck']
    
    dL = gw_data['dL']
    p_pe = gw_data['p_pe']
    nEvents = gw_data['nEvents']
    nsamp = gw_data['nsamp']
    
    z_of_dL = cosmo_funcs['z_of_dL']
    ddL_of_z = cosmo_funcs['ddL_of_z']
    logPriorUniverse = prob_funcs['logPriorUniverse']
    
    # Convert distances to redshifts
    #print(f"Computing redshifts: dL={dL}, H0={H0}, Om0={Om0}")
    z = z_of_dL(dL, H0, Om0)
    
    # Compute log weights
    #print(f"Computing log weights: z={z}, p_pe={p_pe}, nEvents={nEvents}, nsamp={nsamp}")
    log_weights = (
        -jnp.log(ddL_of_z(z, dL, H0, Om0)) 
        - jnp.log(p_pe) 
        + logPriorUniverse(z, samples_ind, f, Om0, gamma_agn, gamma_gal)
    )
    e = time.time()
    #print(f"likelihood call time: {e - s} seconds")
    # Reshape and compute log-likelihood
    #print(f"Reshaping and computing log-likelihood: log_weights={log_weights}, nEvents={nEvents}, nsamp={nsamp}")
    log_weights = log_weights.reshape((nEvents, nsamp))
    ll = jnp.sum(-jnp.log(nsamp) + logsumexp(log_weights, axis=-1))
    
    return ll


def compute_likelihood_grid(
    gw_data, catalog_data, cosmo_funcs, prob_funcs,
    H0_grid, f_grid, Om0=None, gamma_agn=0, gamma_gal=0,
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
    f_grid : array
        1D array of f values to evaluate
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
        2D array of log-likelihood values with shape (len(H0_grid), len(f_grid))
        where log_likelihood[i, j] corresponds to H0_grid[i], f_grid[j]
    """
    print(f"Computing likelihood grid: H0_grid shape={len(H0_grid)}, f_grid shape={len(f_grid)}, Om0={Om0}, gamma_agn={gamma_agn}, gamma_gal={gamma_gal}")
    
    # Precompute pixel indices once (they don't depend on H0, f, Om0, or gammas)
    nside = catalog_data['nside']
    samples_ind = compute_pixel_indices(gw_data['ra'], gw_data['dec'], nside)
    
    if Om0 is None:
        Om0 = cosmo_funcs['Om0Planck']
    
    # Convert to JAX arrays if needed
    H0_grid = jnp.asarray(H0_grid)
    f_grid = jnp.asarray(f_grid)
    
    # Initialize output array
    log_likelihood_grid = jnp.zeros((len(H0_grid), len(f_grid)))
    
    # Compute likelihood for each combination
    if progress:
        total = len(H0_grid) * len(f_grid)
        pbar = tqdm(total=total, desc="Computing likelihood grid")
    
    for i, H0 in enumerate(H0_grid):
        for j, f in enumerate(f_grid):
            ll = compute_darksiren_log_likelihood(
                gw_data, catalog_data, cosmo_funcs, prob_funcs,
                float(H0), float(f), samples_ind, Om0, gamma_agn, gamma_gal
            )
            log_likelihood_grid = log_likelihood_grid.at[i, j].set(float(ll))
            if progress:
                pbar.update(1)
    
    if progress:
        pbar.close()
    
    return np.array(log_likelihood_grid)


def save_likelihood_grid(
    filename, log_likelihood_grid, H0_grid, f_grid,
    config=None, metadata=None, grid_params=None
):
    """
    Save likelihood grid and associated parameters to HDF5 file.
    
    Parameters
    ----------
    filename : str
        Output HDF5 filename
    log_likelihood_grid : array
        2D array of log-likelihood values with shape (len(H0_grid), len(f_grid))
    H0_grid : array
        1D array of H0 values used
    f_grid : array
        1D array of f values used
    config : dict, optional
        Configuration dictionary with:
        - galaxy_file: path to galaxy catalog
        - agn_file: path to AGN catalog
        - gw_file: path to GW samples
        - nside: healpix nside
        - nEvents: number of GW events
        - nsamp: number of samples per event
    metadata : dict, optional
        Additional metadata to save
    grid_params : dict, optional
        Dictionary with grid computation parameters:
        - Om0: matter density parameter
        - gamma_agn: AGN evolution parameter
        - gamma_gal: galaxy evolution parameter
    """
    print(f"Saving likelihood grid to {filename} (shape={log_likelihood_grid.shape})")
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with h5py.File(filename, 'w') as f:
        # Save likelihood grid
        f.create_dataset('log_likelihood_grid', data=np.array(log_likelihood_grid))
        
        # Save parameter grids
        f.create_dataset('H0_grid', data=np.array(H0_grid))
        f.create_dataset('f_grid', data=np.array(f_grid))
        
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
        
        # Save configuration
        if config is not None:
            config_group = f.create_group('config')
            for key, value in config.items():
                if isinstance(value, str):
                    config_group.attrs[key] = value
                elif isinstance(value, (int, float)):
                    config_group.attrs[key] = value
                elif isinstance(value, bool):
                    config_group.attrs[key] = int(value)
                elif isinstance(value, (tuple, list)):
                    # Convert tuples/lists to JSON string
                    config_group.attrs[key] = json.dumps(value)
                elif value is None:
                    config_group.attrs[key] = 'None'
                else:
                    # Convert other types to string
                    config_group.attrs[key] = str(value)
        
        # Save metadata
        if metadata is not None:
            metadata_group = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, str):
                    metadata_group.attrs[key] = value
                elif isinstance(value, (int, float)):
                    metadata_group.attrs[key] = value
                elif isinstance(value, bool):
                    metadata_group.attrs[key] = int(value)
                elif value is None:
                    metadata_group.attrs[key] = 'None'
                else:
                    metadata_group.attrs[key] = str(value)
        
        # Save attributes
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['n_H0'] = len(H0_grid)
        f.attrs['n_f'] = len(f_grid)
        f.attrs['H0_min'] = float(np.min(H0_grid))
        f.attrs['H0_max'] = float(np.max(H0_grid))
        f.attrs['f_min'] = float(np.min(f_grid))
        f.attrs['f_max'] = float(np.max(f_grid))


def load_likelihood_grid(filename):
    """
    Load likelihood grid and associated parameters from HDF5 file.
    
    Parameters
    ----------
    filename : str
        Input HDF5 filename
    
    Returns
    -------
    dict
        Dictionary containing:
        - log_likelihood_grid: 2D array of log-likelihood values
        - H0_grid: 1D array of H0 values
        - f_grid: 1D array of f values
        - grid_params: dictionary with grid computation parameters
        - config: configuration dictionary
        - metadata: metadata dictionary
        - timestamp: timestamp string
    """
    print(f"Loading likelihood grid from {filename}")
    results = {}
    
    with h5py.File(filename, 'r') as f:
        # Load likelihood grid
        results['log_likelihood_grid'] = np.array(f['log_likelihood_grid'])
        
        # Load parameter grids
        results['H0_grid'] = np.array(f['H0_grid'])
        results['f_grid'] = np.array(f['f_grid'])
        
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
        
        # Load configuration
        if 'config' in f:
            config = {}
            for key in f['config'].attrs:
                value = f['config'].attrs[key]
                # Try to convert back to appropriate type
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                if value == 'None':
                    value = None
                # Try to parse as JSON (for tuples/lists)
                elif isinstance(value, str) and (value.startswith('[') or value.startswith('(')):
                    try:
                        value = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        pass
                config[key] = value
            results['config'] = config
        
        # Load metadata
        if 'metadata' in f:
            metadata = {}
            for key in f['metadata'].attrs:
                value = f['metadata'].attrs[key]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                if value == 'None':
                    value = None
                metadata[key] = value
            results['metadata'] = metadata
        
        # Load attributes
        results['timestamp'] = f.attrs.get('timestamp', None)
        results['n_H0'] = f.attrs.get('n_H0', None)
        results['n_f'] = f.attrs.get('n_f', None)
        results['H0_min'] = f.attrs.get('H0_min', None)
        results['H0_max'] = f.attrs.get('H0_max', None)
        results['f_min'] = f.attrs.get('f_min', None)
        results['f_max'] = f.attrs.get('f_max', None)
    
    return results


def setup_mcmc_parameters(
    H0_bounds=(20, 120),
    f_bounds=(0, 1),
    Om0_bounds=None,
    gamma_bounds=(-30, 30)
):
    """
    Set up MCMC parameter bounds and labels.
    
    Parameters
    ----------
    H0_bounds : tuple
        (lower, upper) bounds for H0
    f_bounds : tuple
        (lower, upper) bounds for f
    Om0_bounds : tuple, optional
        (lower, upper) bounds for Om0 (default: from cosmology setup)
    gamma_bounds : tuple
        (lower, upper) bounds for gamma parameters
    
    Returns
    -------
    dict
        Dictionary containing:
        - lower_bound: list of lower bounds
        - upper_bound: list of upper bounds
        - labels: list of parameter labels
        - ndims: number of dimensions
    """
    print(f"Setting up MCMC parameters: H0_bounds={H0_bounds}, f_bounds={f_bounds}")
    lower_bound = [H0_bounds[0], f_bounds[0]]
    upper_bound = [H0_bounds[1], f_bounds[1]]
    
    labels = [r'$H_0$', r'$f$']
    
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
    
    # Precompute pixel indices once (they don't depend on H0, f, Om0, or gammas)
    nside = catalog_data['nside']
    samples_ind = compute_pixel_indices(gw_data['ra'], gw_data['dec'], nside)
    
    def likelihood_emcee(coord):
        """Likelihood function with bounds checking for emcee."""
        for i in range(len(coord)):
            if (coord[i] < lower_bound[i] or coord[i] > upper_bound[i]):
                return -np.inf
        H0, f = coord
        ll = compute_darksiren_log_likelihood(
            gw_data, catalog_data, cosmo_funcs, prob_funcs,
            H0, f, samples_ind, Om0, gamma_agn, gamma_gal
        )
        if np.isnan(ll):
            return -np.inf
        else:
            return float(ll)
    
    return likelihood_emcee


def solve_fagn_lambda(frac_agn_obs, N_gal, N_agn):
    """
    Solve for f_agn and lambda_agn given observed AGN fraction.
    
    Parameters
    ----------
    frac_agn_obs : float
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
        _, frac_agn_model = utils.compute_gw_host_fractions(N_gal, N_agn, f_agn, lambda_agn=lambda_agn)
        return (frac_agn_model - frac_agn_obs)**2
    
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
    likelihood_func, lower_bound, upper_bound, n_walkers=64, n_steps=1000,
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
    n_walkers : int
        Number of walkers (default: 64)
    n_steps : int
        Number of MCMC steps (default: 1000)
    seed : int, optional
        Random seed for initialization
    
    Returns
    -------
    emcee.EnsembleSampler
        Sampler object with chain
    """
    print(f"Running MCMC sampling: n_walkers={n_walkers}, n_steps={n_steps}, seed={seed}, ndims={len(lower_bound)}")
    if emcee is None:
        raise ImportError("emcee is required for MCMC sampling. Install with: pip install emcee")
    
    ndims = len(lower_bound)
    
    if seed is not None:
        np.random.seed(seed)
    
    p0 = np.random.uniform(lower_bound, upper_bound, size=(n_walkers, ndims))
    
    sampler = emcee.EnsembleSampler(
        n_walkers, ndims, likelihood_func,
        moves=[
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ]
    )
    
    sampler.run_mcmc(p0, n_steps, progress=True)
    
    return sampler


def get_posterior_samples(sampler, burnin_frac=0.5, n_samples=None):
    """
    Extract posterior samples from MCMC chain.
    
    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        Sampler object with chain
    burnin_frac : float
        Fraction of chain to discard as burn-in (default: 0.5)
    n_samples : int, optional
        Number of samples to return (default: all post-burnin)
    
    Returns
    -------
    array
        Posterior samples
    """
    print(f"Extracting posterior samples: burnin_frac={burnin_frac}, n_samples={n_samples}")
    shape = sampler.flatchain.shape[0]
    burnin_idx = int(shape * burnin_frac)
    samples = sampler.flatchain[burnin_idx:, :]
    
    if n_samples is not None and n_samples < len(samples):
        choose = np.random.randint(0, len(samples), n_samples)
        samples = samples[choose]
    
    return samples


if __name__ == "__main__":
    main()