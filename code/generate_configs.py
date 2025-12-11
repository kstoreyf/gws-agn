#!/usr/bin/env python3
"""
Script to generate YAML configuration files for generate_gwsamples.py and pixelize_catalogs.py
"""

import yaml
import os
from pathlib import Path

# Try to import astropy for Planck15, fallback to hardcoded values
try:
    from astropy.cosmology import Planck15
    PLANCK15_H0 = Planck15.H0.value  # H0 in km/s/Mpc
    PLANCK15_H = PLANCK15_H0 / 100.0  # Dimensionless Hubble parameter (h = H0/100)
    PLANCK15_OM0 = Planck15.Om0
    PLANCK15_OB0 = Planck15.Ob0  # Baryon density parameter
except ImportError:
    # Fallback values if astropy not available
    PLANCK15_H0 = 67.74  # km/s/Mpc
    PLANCK15_H = PLANCK15_H0 / 100.0
    PLANCK15_OM0 = 0.3075
    PLANCK15_OB0 = 0.0486  # Baryon density parameter


def create_config_data(
    config_name,
    base_path='../data/mocks_glass/mock_seed42_ratioNgalNagn1_bgal1.0_bagn1.0/',
    # Mock catalog parameters
    seed=42,
    nbar_gal=1e-2,
    nbar_agn=1e-2,
    bias_gal=1.0,
    bias_agn=1.0,
    z_min=0.0,
    z_max=1.5,
    nside=256,
    # GW injection parameters
    f_agn=0.25,
    lambda_agn=0.25,
    N_gw=1000,
    gw_seed=None,
    # GW sample generation parameters
    nsamp=10000,
    mass_mean=35.0,
    mass_std=5.0,
    ra_uncertainty=0.01,
    dec_uncertainty=0.01,
    mass_uncertainty=1.5,
    # Cosmology parameters
    h=None,
    H0=None,
    Om0=None,
    Ob0=None,
    output_dir='../configs/configs_data/'
):
    """
    Create a YAML configuration file for data generation and pixelization.
    
    Parameters:
    -----------
    config_name : str
        Name of the config file (without .yaml extension)
    base_path : str
        Base path for input/output data files
    seed : int
        Seed for mock generation
    nbar_gal : float
        Number density of galaxies
    nbar_agn : float
        Number density of AGN
    bias_gal : float
        Bias parameter for galaxies
    bias_agn : float
        Bias parameter for AGNs
    z_min : float
        Minimum redshift
    z_max : float
        Maximum redshift
    nside : int
        HEALPix resolution parameter
    f_agn : float
        Fraction of AGNs hosting GW events
    lambda_agn : float
        Lambda parameter for AGN GW selection
    N_gw : int
        Number of GW events to inject
    gw_seed : int, optional
        Seed for GW injection (None = seed + 1000)
    nsamp : int
        Number of samples per GW event
    mass_mean : float
        Mean black hole mass in solar masses
    mass_std : float
        Standard deviation of black hole mass
    ra_uncertainty : float
        RA uncertainty in radians
    dec_uncertainty : float
        Dec uncertainty in radians
    mass_uncertainty : float
        Mass uncertainty in solar masses
    h : float, optional
        Dimensionless Hubble parameter (defaults to Planck 2015 value)
    H0 : float, optional
        Hubble constant in km/s/Mpc (defaults to Planck 2015 value, should equal 100*h)
    Om0 : float, optional
        Matter density parameter (defaults to Planck 2015 value)
    Ob0 : float, optional
        Baryon density parameter (defaults to Planck 2015 value)
    output_dir : str
        Directory to save config file
    """
    # Use Planck 2015 defaults if not specified
    if h is None:
        h = PLANCK15_H
    if H0 is None:
        H0 = PLANCK15_H0
    if Om0 is None:
        Om0 = PLANCK15_OM0
    if Ob0 is None:
        Ob0 = PLANCK15_OB0
    # Get absolute path for output directory (relative to script location)
    script_dir = Path(__file__).parent.parent  # Go up from code/ to project root
    if not os.path.isabs(output_dir):
        # Handle relative paths like '../configs/configs_data/'
        if output_dir.startswith('../'):
            output_dir = str(script_dir / output_dir[3:])  # Remove '../' prefix
        else:
            output_dir = str(script_dir / output_dir)
    # else: output_dir is already absolute
    
    # Construct filenames based on parameters
    # Calculate ratio for path construction
    ratio_ngal_nagn = int(round(nbar_gal / nbar_agn))
    tag_mock_extra = f'_bgal{bias_gal}_bagn{bias_agn}'
    tag_mock = f'_seed{seed}_ratioNgalNagn{ratio_ngal_nagn}{tag_mock_extra}'
    
    # Determine gw_seed for filename if None
    gw_seed_for_filename = gw_seed if gw_seed is not None else seed + 1000
    
    mock_catalog = 'mock_catalog.h5'
    mock_gw_indices = f'gws_fagn{f_agn}_lambdaagn{lambda_agn}_N{N_gw}_seed{gw_seed_for_filename}.h5'
    gw_samples_output = f'gwsamples_fagn{f_agn}_lambdaagn{lambda_agn}_N{N_gw}_seed{gw_seed_for_filename}.h5'
    pixelated_galaxies = f'lognormal_pixelated_nside_{nside}_galaxies.h5'
    pixelated_agn = f'lognormal_pixelated_nside_{nside}_agn.h5'
    
    # Create config dictionary
    config = {
        'mock_catalog': {
            'seed': seed,
            'nbar_gal': nbar_gal,
            'nbar_agn': nbar_agn,
            'bias_gal': bias_gal,
            'bias_agn': bias_agn,
            'z_min': z_min,
            'z_max': z_max,
            'nside': nside
        },
        'gw_injection': {
            'f_agn': f_agn,
            'lambda_agn': lambda_agn,
            'N_gw': N_gw,
            'gw_seed': gw_seed
        },
        'paths': {
            'base_path': base_path,
            'mock_catalog': mock_catalog,
            'mock_gw_indices': mock_gw_indices,
            'gw_samples_output': gw_samples_output,
            'pixelated_galaxies': pixelated_galaxies,
            'pixelated_agn': pixelated_agn
        },
        'gw_samples': {
            'nsamp': nsamp,
            'mass_mean': mass_mean,
            'mass_std': mass_std,
            'ra_uncertainty': ra_uncertainty,
            'dec_uncertainty': dec_uncertainty,
            'mass_uncertainty': mass_uncertainty
        },
        'cosmology': {
            'h': h,
            'H0': H0,
            'Om0': Om0,
            'Ob0': Ob0
        },
        'pixelization': {
            'nside': nside
        }
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write YAML file
    output_path = os.path.join(output_dir, f'{config_name}.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created config file: {output_path}")
    return output_path


def main():
    """
    Main function to create config files.
    Modify the parameters below to create different configs.
    """
    # Example: Create a default config
    create_config_data(
        config_name='config_data_default',
        base_path='../data/mocks_glass/mock_seed42_ratioNgalNagn1_bgal1.0_bagn1.0/',
        seed=42,
        nbar_gal=1e-2,
        nbar_agn=1e-2,
        bias_gal=1.0,
        bias_agn=1.0,
        z_min=0.0,
        z_max=1.5,
        nside=256,
        f_agn=0.25,
        lambda_agn=0.25,
        N_gw=1000,
        gw_seed=None,  # Will default to seed + 1000
        nsamp=10000,
        # Cosmology will default to Planck 2015 values
        mass_mean=35.0,
        mass_std=5.0,
        ra_uncertainty=0.01,
        dec_uncertainty=0.01,
        mass_uncertainty=1.5
    )
    
    # Add more configs here as needed
    # Example: Create a high-resolution config
    # create_config_data(
    #     config_name='config_data_highres',
    #     nside=512,
    #     nsamp=20000,
    #     ...
    # )


def create_config_inference(
    config_name,
    config_data_path='../configs/configs_data/config_data_default.yaml',
    method='mcmc',
    # MCMC parameters
    n_walkers=32,
    n_steps=10000,
    burnin_frac=0.2,
    mcmc_seed=None,
    # Likelihood grid parameters
    n_H0=20,
    n_f=20,
    # Parameter bounds
    H0_bounds=[50, 100],
    f_bounds=[0, 1],
    Om0_bounds=None,
    gamma_bounds=[-30, 30],
    # Cosmology parameters
    Om0=None,
    gamma_agn=0,
    gamma_gal=0,
    # Output settings
    output_dir='../results/inference/',
    tag_inf_extra='_norm',
    output_config_dir='../configs/configs_inference/'
):
    """
    Create a YAML configuration file for inference.
    
    Parameters:
    -----------
    config_name : str
        Name of the config file (without .yaml extension)
    config_data_path : str
        Path to the data configuration file (relative to project root)
    method : str
        Inference method: 'mcmc' or 'likelihood_grid'
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    burnin_frac : float
        Fraction of chain to discard as burn-in
    mcmc_seed : int, optional
        Random seed for MCMC initialization
    n_H0 : int
        Number of H0 grid points for likelihood_grid
    n_f : int
        Number of f grid points for likelihood_grid
    H0_bounds : list
        [lower, upper] bounds for H0
    f_bounds : list
        [lower, upper] bounds for f
    Om0_bounds : list, optional
        [lower, upper] bounds for Om0
    gamma_bounds : list
        [lower, upper] bounds for gamma parameters
    Om0 : float, optional
        Matter density parameter (None = use Planck default)
    gamma_agn : float
        AGN evolution parameter
    gamma_gal : float
        Galaxy evolution parameter
    output_dir : str
        Directory for output files
    tag_inf_extra : str
        Extra suffix for inference output (replaces output_suffix)
    output_config_dir : str
        Directory to save config file
    """
    # Get absolute path for output directory
    script_dir = Path(__file__).parent.parent
    if not os.path.isabs(output_config_dir):
        if output_config_dir.startswith('../'):
            output_config_dir = str(script_dir / output_config_dir[3:])
        else:
            output_config_dir = str(script_dir / output_config_dir)
    
    # Load config_data to extract parameters for filename construction
    if not os.path.isabs(config_data_path):
        if config_data_path.startswith('../'):
            config_data_abs_path = str(script_dir / config_data_path[3:])
        else:
            config_data_abs_path = str(script_dir / config_data_path)
    else:
        config_data_abs_path = config_data_path
    
    with open(config_data_abs_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Extract parameters from config_data for filename construction
    seed = config_data['mock_catalog']['seed']
    nside = config_data['pixelization']['nside']
    bias_gal = config_data['mock_catalog']['bias_gal']
    bias_agn = config_data['mock_catalog']['bias_agn']
    nbar_gal = config_data['mock_catalog']['nbar_gal']
    nbar_agn = config_data['mock_catalog']['nbar_agn']
    ratioNgalNagn = int(round(nbar_gal / nbar_agn))
    f_agn = config_data['gw_injection']['f_agn']
    lambda_agn = config_data['gw_injection']['lambda_agn']
    N_gw = config_data['gw_injection']['N_gw']
    gw_seed = config_data['gw_injection']['gw_seed']
    if gw_seed is None:
        gw_seed = seed + 1000
    
    # Build tags in structured format
    tag_cat = f'_seed{seed}_ratioNgalNagn{ratioNgalNagn}_bgal{bias_gal}_bagn{bias_agn}'
    tag_gw = f'_fagn{f_agn}_lambdaagn{lambda_agn}'
    
    # Build tag_inf (inference-specific)
    if method == 'mcmc':
        tag_inf = f'_mcmc_nw{n_walkers}_ns{n_steps}'
    elif method == 'likelihood_grid':
        tag_inf = f'_grid_nH0_{n_H0}_nf_{n_f}'
    else:
        tag_inf = ''
    
    # tag_inf_extra is the extra suffix parameter
    
    # Build full tag for output filename
    tag_full = f'{tag_cat}{tag_gw}_N{N_gw}_seed{gw_seed}_nside{nside}{tag_inf}{tag_inf_extra}'
    
    # Build full output filename
    fn_inf = os.path.join(output_dir, f'inference_results{tag_full}.h5')
    
    # Create config dictionary
    config = {
        'config_data_path': config_data_path,
        'method': method,
        'mcmc': {
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'burnin_frac': burnin_frac,
            'seed': mcmc_seed
        },
        'likelihood_grid': {
            'n_H0': n_H0,
            'n_f': n_f
        },
        'parameter_bounds': {
            'H0_bounds': H0_bounds,
            'f_bounds': f_bounds,
            'Om0_bounds': Om0_bounds,
            'gamma_bounds': gamma_bounds
        },
        'cosmology': {
            'Om0': Om0,
            'gamma_agn': gamma_agn,
            'gamma_gal': gamma_gal
        },
        'output': {
            'output_dir': output_dir,
            'tag_cat': tag_cat,
            'tag_gw': tag_gw,
            'tag_inf': tag_inf,
            'tag_inf_extra': tag_inf_extra,
            'tag_full': tag_full,
            'fn_inf': fn_inf
        }
    }
    
    # Ensure output directory exists
    os.makedirs(output_config_dir, exist_ok=True)
    
    # Write YAML file
    output_path = os.path.join(output_config_dir, f'{config_name}.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created inference config file: {output_path}")
    return output_path


def main_inference():
    """
    Main function to create inference config files.
    Modify the parameters below to create different configs.
    """
    # Example: Create an MCMC config
    create_config_inference(
        config_name='config_inference_mcmc_default',
        config_data_path='../configs/configs_data/config_data_default.yaml',
        method='mcmc',
        n_walkers=32,
        n_steps=10000,
        burnin_frac=0.2,
        mcmc_seed=None,
        H0_bounds=[50, 100],
        f_bounds=[0, 1],
        tag_inf_extra='_norm'
    )
    
    # Example: Create a likelihood grid config
    create_config_inference(
        config_name='config_inference_grid_default',
        config_data_path='../configs/configs_data/config_data_default.yaml',
        method='likelihood_grid',
        n_H0=20,
        n_f=20,
        H0_bounds=[50, 100],
        f_bounds=[0, 1],
        tag_inf_extra='_norm'
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--inference':
        # Create inference configs
        main_inference()
    else:
        # Create data configs
        main()
