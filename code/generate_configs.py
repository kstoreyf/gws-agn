#!/usr/bin/env python3
"""
Script to generate YAML configuration files for generate_gwsamples.py and pixelize_catalogs.py
"""

import yaml
import os

# Try to import astropy for Planck15, fallback to hardcoded values
try:
    from astropy.cosmology import Planck15
    PLANCK15_H0 = float(Planck15.H0.value)  # H0 in km/s/Mpc (convert to native float)
    PLANCK15_H = float(PLANCK15_H0 / 100.0)  # Dimensionless Hubble parameter (h = H0/100)
    PLANCK15_OM0 = float(Planck15.Om0)  # Convert to native float
    PLANCK15_OB0 = float(Planck15.Ob0)  # Baryon density parameter (convert to native float)
except ImportError:
    # Fallback values if astropy not available
    PLANCK15_H0 = 67.74  # km/s/Mpc
    PLANCK15_H = 0.6774  # Dimensionless Hubble parameter
    PLANCK15_OM0 = 0.3075
    PLANCK15_OB0 = 0.0486  # Baryon density parameter


def create_config_data(
    fn_config=None,
    dir_mock=None,  # Will be auto-generated from tag_cat
    # Mock catalog parameters
    seed=42,
    nbar_gal=1e-2,
    nbar_agn=1e-2,
    bias_gal=1.0,
    bias_agn=1.0,
    z_min=0.0,
    z_max=1.5,
    nside=256,
    gamma_agn=0.0,
    gamma_gal=0.0,
    # GW injection parameters
    f_agn=0.25,
    lambda_agn=0.25,
    N_gw=1000,
    seed_gw=None,
    z_max_gw=None,
    # GW sample generation parameters
    N_samples_gw=10000,
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
    dir_configs='../configs/configs_data/',
    overwrite_config=False
):
    """
    Create a YAML configuration file for data generation and pixelization.
    
    Parameters:
    -----------
    fn_config : str, optional
        Full path to config file (directory + filename + .yaml). If None, auto-generated from tags.
    dir_mock : str
        Directory for mock catalog data files
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
    seed_gw : int, optional
        Seed for GW injection (None = seed + 1000)
    z_max_gw : float, optional
        Maximum redshift for GW host selection. If None, uses z_max (full catalog range).
    N_samples_gw : int
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
    dir_configs : str
        Directory to save config file
    overwrite_config : bool
        If True, overwrite existing config file. If False, skip if file exists.
    """
    # Use Planck 2015 defaults if not specified
    # Convert all to native Python floats to avoid numpy serialization issues in YAML
    h = float(PLANCK15_H if h is None else h)
    H0 = float(PLANCK15_H0 if H0 is None else H0)
    Om0 = float(PLANCK15_OM0 if Om0 is None else Om0)
    Ob0 = float(PLANCK15_OB0 if Ob0 is None else Ob0)
    
    # Validate h and H0 consistency
    expected_H0_from_h = 100.0 * h
    if abs(H0 - expected_H0_from_h) > 0.01:  # Allow small floating point differences
        raise ValueError(
            f"Inconsistent cosmology parameters: h={h} implies H0={expected_H0_from_h:.2f} km/s/Mpc, " \
            f"but H0={H0:.2f} km/s/Mpc was provided. They must satisfy H0 = 100 * h."
        )
    
    # Build tags
    ratio_ngal_nagn = int(round(nbar_gal / nbar_agn))
    tag_cat = f'_seed{seed}_ratioNgalNagn{ratio_ngal_nagn}_bgal{bias_gal}_bagn{bias_agn}'
    tag_pix = f'_nside{nside}'
    tag_gw = f'_fagn{f_agn}_lambdaagn{lambda_agn}'
    if z_max_gw is not None:
        tag_gw += f'_zmaxgw{z_max_gw}'
    
    # Construct dir_mock using tag_cat (auto-generate if not provided)
    if dir_mock is None:
        dir_mock = f'../data/mocks_glass/mock{tag_cat}/'
    
    # Auto-generate fn_config from tags if not provided
    if fn_config is None:
        fn_config = os.path.join(dir_configs, f'config_data{tag_cat}{tag_pix}{tag_gw}.yaml')
    
    # Construct filenames using tags
    name_cat = 'mock_catalog.h5'
    name_gw = f'gws{tag_gw}.h5'
    name_gwsamples = f'gwsamples{tag_gw}.h5'
    name_cat_gal_pixelated = f'cat_gal_pixelated_nside{nside}.h5'
    name_cat_agn_pixelated = f'cat_agn_pixelated_nside{nside}.h5'
    
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
            'nside': nside,
            'gamma_agn': float(gamma_agn),
            'gamma_gal': float(gamma_gal)
        },
        'gw_injection': {
            'f_agn': f_agn,
            'lambda_agn': lambda_agn,
            'N_gw': N_gw,
            'seed_gw': seed_gw,
            'z_max_gw': z_max_gw if z_max_gw is not None else z_max
        },
        'paths': {
            'dir_mock': dir_mock,
            'name_cat': name_cat,
            'name_gw': name_gw,
            'name_gwsamples': name_gwsamples,
            'name_cat_gal_pixelated': name_cat_gal_pixelated,
            'name_cat_agn_pixelated': name_cat_agn_pixelated,
            'tag_cat': tag_cat,
            'tag_pix': tag_pix,
            'tag_gw': tag_gw
        },
        'gw_samples': {
            'N_samples_gw': N_samples_gw,
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
    os.makedirs(os.path.dirname(fn_config), exist_ok=True)
    
    # Check if file exists and overwrite_config flag
    if os.path.exists(fn_config) and not overwrite_config:
        print(f"Config file already exists (skipping): {fn_config}")
        return fn_config
    
    with open(fn_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created config file: {fn_config}")
    
    return fn_config


def main_data(overwrite_config=False):
    """
    Main function to create config files.
    Modify the parameters below to create different configs.
    
    Parameters
    ----------
    overwrite_config : bool
        If True, overwrite existing config files. If False, skip if files exist.
    """
    # Example: Create a default config (fn_config and dir_mock will be auto-generated from tags)
    create_config_data(
        fn_config=None,  # Auto-generate from tags
        dir_mock=None,  # Auto-generate from tag_cat
        seed=42,
        nbar_gal=1e-2,
        nbar_agn=1e-2,
        bias_gal=1.0,
        bias_agn=1.0,
        z_min=0.0,
        z_max=1.5,
        #nside=256,
        nside=64,
        f_agn=0.0,
        lambda_agn=0.0,
        N_gw=1000,
        seed_gw=1042,
        z_max_gw=1.0,
        N_samples_gw=10000,
        # Cosmology will default to Planck 2015 values
        mass_mean=35.0,
        mass_std=5.0,
        ra_uncertainty=0.01,
        dec_uncertainty=0.01,
        mass_uncertainty=1.5,
        overwrite_config=overwrite_config
    )


def create_config_inference(
    fn_config=None,
    fn_config_data=None,
    mode_inf='mcmc',
    # MCMC parameters
    N_walkers=32,
    N_steps=10000,
    burnin_frac=0.2,
    seed_mcmc=0,
    # Likelihood grid parameters
    N_H0=20,
    N_alpha_agn=20,
    N_gw_inf=None,
    # Parameter roles
    parameters_vary=None,
    parameters_fix=None,
    # Catalog redshift uncertainty
    Dz_gal=0.0001,
    Dz_agn=0.0001,
    # Output settings
    dir_inference='../results/inference/',
    tag_inf_extra='',
    dir_configs='../configs/configs_inference/',
    overwrite_config=False
):
    """
    Create a YAML configuration file for inference.
    
    Parameters:
    -----------
    fn_config : str, optional
        Full path to config file (directory + filename + .yaml). If None, auto-generated from tags.
    fn_config_data : str
        Path to the data configuration file (relative to project root)
    mode_inf : str
        Inference mode: 'mcmc' or 'grid'
    N_walkers : int
        Number of MCMC walkers
    N_steps : int
        Number of MCMC steps
    burnin_frac : float
        Fraction of chain to discard as burn-in
    seed_mcmc : int, optional
        Random seed for MCMC initialization
    N_H0 : int
        Number of H0 grid points for grid
    N_alpha_agn : int
        Number of alpha_agn grid points for grid
    parameters_vary : list of str, optional
        Names of parameters to vary in the inference (subset of
        ['H0', 'alpha_agn', 'Om0', 'gamma_agn', 'gamma_gal']).
        Defaults to ['alpha_agn'].
    parameters_fix : list of str, optional
        Names of parameters to fix (complement of parameters_vary).
        Defaults to ['H0', 'Om0', 'gamma_agn', 'gamma_gal'].
    Dz_gal : float, optional
        Fractional redshift uncertainty for galaxies: sigma_z = Dz_gal * (1 + z).
        Default: 0.0001 (spectroscopic quality).
    Dz_agn : float, optional
        Fractional redshift uncertainty for AGN: sigma_z = Dz_agn * (1 + z).
        Default: 0.0001 (spectroscopic quality).
    dir_inference : str
        Directory for inference output files
    tag_inf_extra : str
        Extra suffix for inference output (replaces output_suffix)
    dir_configs : str
        Directory to save config file
    """
    # Load config_data to extract parameters for filename construction
    with open(fn_config_data, 'r') as f:
        config_data = yaml.safe_load(f)
    
    tag_cat = config_data['paths']['tag_cat']
    tag_pix = config_data['paths']['tag_pix']
    tag_gw = config_data['paths']['tag_gw']
    
    # Default parameter role lists if not provided
    if parameters_vary is None:
        parameters_vary = ['alpha_agn']
    if parameters_fix is None:
        parameters_fix = ['H0', 'Om0', 'gamma_agn', 'gamma_gal']
    
    # Build tag_inf (inference-specific)
    if mode_inf == 'mcmc':
        tag_inf = f'_mcmc_nw{N_walkers}_nsteps{N_steps}'
    elif mode_inf == 'grid':
        tag_inf = f'_grid_nH0{N_H0}_nalphaagn{N_alpha_agn}'
    else:
        tag_inf = ''
    
    # Build full tag for output filename
    tag_full = f'{tag_cat}{tag_pix}{tag_gw}{tag_inf}{tag_inf_extra}'
    
    # Auto-generate fn_config from tags if not provided
    if fn_config is None:
        fn_config = os.path.join(dir_configs, f'config_inference{tag_full}.yaml')
    
    # Build full output filename
    fn_inf = os.path.join(dir_inference, f'inference_results{tag_full}.h5')
    
    # Create config dictionary
    config = {
        'fn_config_data': fn_config_data,
        'mode_inf': mode_inf,
        'N_gw_inf': N_gw_inf,
        'mcmc': {
            'N_walkers': N_walkers,
            'N_steps': N_steps,
            'burnin_frac': burnin_frac,
            'seed_mcmc': seed_mcmc
        },
        'grid': {
            'N_H0': N_H0,
            'N_alpha_agn': N_alpha_agn
        },
        'parameters': {
            'parameters_vary': list(parameters_vary),
            'parameters_fix': list(parameters_fix),
        },
        'catalog': {
            'Dz_gal': Dz_gal,
            'Dz_agn': Dz_agn,
        },
        'paths': {
            'dir_inference': dir_inference,
            'tag_cat': tag_cat,
            'tag_pix': tag_pix,
            'tag_gw': tag_gw,
            'tag_inf': tag_inf,
            'tag_inf_extra': tag_inf_extra,
            'tag_full': tag_full,
            'fn_inf': fn_inf
        }
    }
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(fn_config), exist_ok=True)
    os.makedirs(dir_inference, exist_ok=True)
    
    # Check if file exists and overwrite_config flag
    if os.path.exists(fn_config) and not overwrite_config:
        print(f"Inference config file already exists (skipping): {fn_config}")
        return fn_config
    
    with open(fn_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created inference config file: {fn_config}")
    
    return fn_config


def main_inference(overwrite_config=False):
    """
    Main function to create inference config files.
    Modify the parameters below to create different configs.
    
    Parameters
    ----------
    overwrite_config : bool
        If True, overwrite existing config files. If False, skip if files exist.
    """
    
    # build config_data name
    tag_cat = f'_seed42_ratioNgalNagn1_bgal1.0_bagn1.0'
    tag_gw = f'_fagn0.0_lambdaagn0.0_zmaxgw1.0'
    tag_pix = f'_nside64'
    config_data_name = f'config_data{tag_cat}{tag_pix}{tag_gw}'
    # Now create inference configs referencing the data config
    # create_config_inference(
    #     fn_config=None,  # Auto-generate from tags
    #     fn_config_data=f'../configs/configs_data/{config_data_name}.yaml',
    #     mode_inf='mcmc',
    #     N_walkers=32,
    #     N_steps=500,
    #     burnin_frac=0.2,
    #     seed_mcmc=0,
    #     parameters_vary=['alpha_agn'],
    #     parameters_fix=['H0', 'Om0', 'gamma_agn', 'gamma_gal'],
    #     tag_inf_extra='_vary-alphaagn',
    #     overwrite_config=overwrite_config
    # )
    # Inference config with alpha_agn fixed to zero and H0 free
    create_config_inference(
        fn_config=None,  # Auto-generate from tags
        fn_config_data=f'../configs/configs_data/{config_data_name}.yaml',
        mode_inf='mcmc',
        N_walkers=32,
        N_steps=500,
        burnin_frac=0.2,
        seed_mcmc=0,
        Dz_gal=0.03,
        Dz_agn=0.03,
        parameters_vary=['H0'],
        parameters_fix=['alpha_agn', 'Om0', 'gamma_agn', 'gamma_gal'],
        tag_inf_extra='_Dz0.03_betaH0try3_vary-H0',
        overwrite_config=overwrite_config
    )
    
    # Example: Create a likelihood grid config (fn_config will be auto-generated from tags)
    # create_config_inference(
    #     fn_config=None,  # Auto-generate from tags
    #     fn_config_data=f'../configs/configs_data/{config_data_name}.yaml',
    #     mode_inf='grid',
    #     N_H0=30,
    #     N_alpha_agn=30,
    #     N_gw_inf=None,
    #     tag_inf_extra='_norm5',
    #     overwrite_config=overwrite_config
    # )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate configuration files for GW-AGN pipeline')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data', '-d', action='store_true',
                       help='Generate data configs')
    group.add_argument('--inference', '-i', action='store_true',
                       help='Generate inference configs')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing config files if they exist')
    
    args = parser.parse_args()
    
    overwrite_config = args.overwrite
    
    if args.inference:
        main_inference(overwrite_config=overwrite_config)
    else:
        main_data(overwrite_config=overwrite_config)
