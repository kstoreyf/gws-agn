#!/usr/bin/env python3
"""
Mock galaxy and AGN catalog generation using GLASS.

This script generates mock galaxy and AGN catalogs using the GLASS (Gaussian Lognormal 
Astronomical Simulation Software) package. It creates matter fields, populates them 
with tracers (galaxies and AGN), and saves the results to an HDF5 file.

The script is now separated into two main functions:
1. create_mock_catalog() - Creates the base mock catalog (galaxies and AGN)
2. inject_gw_sources() - Injects GW sources into an existing mock catalog

This allows for multiple GW injection sets from the same base catalog.

Note: You'll need to install glass (version 2025.1), camb, and glass.ext.camb
(literally 'pip install glass.ext.camb')
"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo_astropy
import camb
from cosmology import Cosmology
import glass
import glass.ext.camb
import h5py
import sys
import os
import argparse
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))
import utils


def parse_args():
    """
    Parse command line arguments for config file.
    
    Returns:
    --------
    config : dict
        Configuration dictionary loaded from YAML file
    overwrite_mock : bool
        Whether to overwrite existing mock catalog
    overwrite_gws : bool
        Whether to overwrite existing GW injection
    """
    parser = argparse.ArgumentParser(description='Generate mock galaxy and AGN catalogs')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--overwrite-mock', action='store_true',
                        help='Overwrite existing mock catalog if it exists')
    parser.add_argument('--overwrite-gws', action='store_true',
                        help='Overwrite existing GW injection if it exists')
    args = parser.parse_args()
    
    # Load YAML config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    return config, args.overwrite_mock, args.overwrite_gws


def main(config, overwrite_mock=False, overwrite_gws=False):
    """Main function to create mock catalog and inject GW sources."""
    
    # Extract parameters from config
    seed = config['mock_catalog']['seed']
    nbar_gal = config['mock_catalog']['nbar_gal']
    nbar_agn = config['mock_catalog']['nbar_agn']
    bias_gal = config['mock_catalog']['bias_gal']
    bias_agn = config['mock_catalog']['bias_agn']
    z_min = config['mock_catalog']['z_min']
    z_max = config['mock_catalog']['z_max']
    nside = config['mock_catalog']['nside']
    lmax = nside  # Set lmax equal to nside (as in original code)
    
    # GW injection parameters
    f_agn = config['gw_injection']['f_agn']
    lambda_agn = config['gw_injection']['lambda_agn']
    N_gw = config['gw_injection']['N_gw']
    gw_seed = config['gw_injection']['gw_seed']  # Can be None
    
    # Generate filename for the mock catalog
    tag_mock_extra = f'_bgal{bias_gal}_bagn{bias_agn}'
    tag_mock = f'_seed{seed}_ratioNgalNagn{int(round(nbar_gal/nbar_agn))}{tag_mock_extra}'
    dir_mock = f'../data/mocks_glass/mock{tag_mock}'
    fn_mock = os.path.join(dir_mock, 'mock_catalog.h5')
    
    print("=== Creating Mock Catalog ===")
    ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, catalog_attrs = create_mock_catalog(
        seed=seed, nbar_gal=nbar_gal, nbar_agn=nbar_agn, 
        bias_gal=bias_gal, bias_agn=bias_agn,
        z_min=z_min, z_max=z_max, nside=nside, lmax=lmax,
        fn_mock=fn_mock, save=True, overwrite_mock=overwrite_mock
    )
    
    print("\n=== Injecting GW Sources ===")

    i_gw_gal, i_gw_agn, N_gw, f_agn, lambda_agn, gw_seed = inject_gw_sources(
        fn_mock, f_agn=f_agn, N_gw=N_gw, gw_seed=gw_seed, lambda_agn=lambda_agn,
        save=True, overwrite_gws=overwrite_gws
    )
    
    print("\nMock catalog generation and GW injection complete!")


def compute_3d_positions(lon, lat, redshift):
    """Compute 3D Cartesian positions from lon, lat, and redshift arrays."""
    x = redshift * np.cos(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
    y = redshift * np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
    z = redshift * np.sin(np.deg2rad(lat))
    return np.stack([x, y, z], axis=-1)


def save_mock_catalog(fn_mock, ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn,
                     N_gal, N_agn, bias_gal, bias_agn, z_max, nside, seed, h, Oc, Ob):
    """Save mock catalog data to HDF5 file (without GW sources)."""
    compression = 'gzip'
    compression_opts = 9
    with h5py.File(fn_mock, 'w') as f:
        # Create datasets with compression
        f.create_dataset('ra_gal', data=ra_gal, compression=compression, compression_opts=compression_opts)
        f.create_dataset('dec_gal', data=dec_gal, compression=compression, compression_opts=compression_opts)
        f.create_dataset('z_gal', data=z_gal, compression=compression, compression_opts=compression_opts)
        f.create_dataset('ra_agn', data=ra_agn, compression=compression, compression_opts=compression_opts)
        f.create_dataset('dec_agn', data=dec_agn, compression=compression, compression_opts=compression_opts)
        f.create_dataset('z_agn', data=z_agn, compression=compression, compression_opts=compression_opts)
        
        # attributes of sim
        f.attrs['n_gal'] = N_gal
        f.attrs['n_agn'] = N_agn
        f.attrs['b1_gal'] = bias_gal
        f.attrs['b1_agn'] = bias_agn
        f.attrs['z_max'] = z_max
        f.attrs['nside'] = nside
        f.attrs['seed'] = seed
        
        # cosmology
        f.attrs['h'] = h
        f.attrs['Oc'] = Oc
        f.attrs['Ob'] = Ob
    
    print(f"Mock catalog saved to {fn_mock}")


def load_mock_catalog(fn_mock):
    """Load mock catalog from HDF5 file."""
    with h5py.File(fn_mock, 'r') as f:
        # Load position data
        ra_gal = f['ra_gal'][:]
        dec_gal = f['dec_gal'][:]
        z_gal = f['z_gal'][:]
        ra_agn = f['ra_agn'][:]
        dec_agn = f['dec_agn'][:]
        z_agn = f['z_agn'][:]
        
        # Load attributes
        attrs = dict(f.attrs)
    
    return ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, attrs


def save_gw_injection(fn_gw, i_gw_gal, i_gw_agn, N_gw, f_agn, lambda_agn, gw_seed):
    """Save GW injection data to HDF5 file."""
    compression = 'gzip'
    compression_opts = 9
    with h5py.File(fn_gw, 'w') as f:
        # Create datasets with compression
        f.create_dataset('i_gw_gal', data=i_gw_gal, compression=compression, compression_opts=compression_opts)
        f.create_dataset('i_gw_agn', data=i_gw_agn, compression=compression, compression_opts=compression_opts)
        
        # attributes
        f.attrs['n_gw'] = N_gw
        f.attrs['f_agn'] = f_agn
        f.attrs['lambda_agn'] = lambda_agn
        f.attrs['gw_seed'] = gw_seed
    
    print(f"GW injection saved to {fn_gw}")


def load_gw_injection(fn_gw):
    """Load GW injection data from HDF5 file."""
    with h5py.File(fn_gw, 'r') as f:
        # Load GW source indices
        i_gw_gal = f['i_gw_gal'][:]
        i_gw_agn = f['i_gw_agn'][:]
        
        # Load attributes
        attrs = dict(f.attrs)
    
    return i_gw_gal, i_gw_agn, attrs


def create_mock_catalog(seed=42, nbar_gal=1e-1, nbar_agn=1e-3, bias_gal=1.5, bias_agn=2.5,
                       z_min=0.0, z_max=1.5, nside=128, lmax=128, fn_mock=None, save=True, overwrite_mock=False):
    """Create a mock galaxy and AGN catalog."""
    
    # Create output directory if it doesn't exist
    dir_mock = os.path.dirname(fn_mock)
    os.makedirs(dir_mock, exist_ok=True)
    
    # Check if mock already exists
    if os.path.exists(fn_mock) and not overwrite_mock:
        print(f"Mock catalog already exists: {fn_mock}")
        print("Loading existing catalog...")
        return load_mock_catalog(fn_mock)
    
    if os.path.exists(fn_mock) and overwrite_mock:
        print(f"Mock catalog exists but overwrite_mock=True, regenerating: {fn_mock}")
    else:
        print("Creating new mock catalog...")
    
    # Creating a numpy random number generator for sampling
    rng = np.random.default_rng(seed=seed)
    
    # Cosmology for the simulation
    h = cosmo_astropy.h
    Oc = cosmo_astropy.Om0 - cosmo_astropy.Ob0
    Ob = cosmo_astropy.Ob0
    print(f"Using cosmology: h = {h}, Oc = {Oc}, Ob = {Ob}")
    
    # Set up CAMB parameters for matter angular power spectrum
    pars = camb.set_params(
        H0=100 * h,
        omch2=Oc * h**2,
        ombh2=Ob * h**2,
        NonLinear=camb.model.NonLinear_both,
    )
    
    # Get the cosmology from CAMB
    cosmo = Cosmology.from_camb(pars)
    
    # Generate matter fields
    print("Generating matter fields...")
    
    # Shells of 200 Mpc in comoving distance spacing
    zb = glass.distance_grid(cosmo, z_min, z_max, dx=200.0)
    
    # Linear radial window functions
    shells = glass.linear_windows(zb)
    
    # Compute the angular matter power spectra of the shells with CAMB
    cls = glass.ext.camb.matter_cls(pars, lmax, shells)
    
    # Set up lognormal matter fields for simulation
    fields = glass.lognormal_fields(shells)
    
    # Apply discretisation to the full set of spectra
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)
    
    # Compute Gaussian spectra for lognormal fields from discretised spectra
    gls = glass.solve_gaussian_spectra(fields, cls)
    
    # Generator for lognormal matter fields
    matter = glass.generate(fields, gls, nside, ncorr=3, rng=rng)
    
    # Create volume-weighted dN/dz
    z_bins = np.linspace(z_min, z_max, 100)
    volume_weights = glass.volume_weight(z_bins, cosmo)
    dndz = volume_weights / np.max(volume_weights)
    
    # Generate galaxies
    print("Generating galaxies...")
    dndz_gal = nbar_gal * dndz  # volume-weighted
    ngal_arr = glass.partition(z_bins, dndz_gal, shells)
    
    # Generate AGN
    print("Generating AGN...")
    dndz_agn = nbar_agn * dndz  # volume-weighted
    nagn_arr = glass.partition(z_bins, dndz_agn, shells)
        
    # Generate tracer positions
    print("Generating tracer positions...")
    tracers = {
        'galaxies': {'bias': bias_gal, 'N': ngal_arr},
        'agn': {'bias': bias_agn, 'N': nagn_arr},
    }
    
    positions_3d = {name: [] for name in tracers}
    positions_sky = {name: [] for name in tracers}  # Will hold (ra, dec, z)
    
    matter = list(matter)  # Convert generator to a list for multiple iterations
    for tracer_name, tracer_dict in tracers.items():
        print(f"Processing tracer: {tracer_name}")
        for i, delta_i in enumerate(matter):
            print(f"Processing shell: {i}")
            # Get all positions for this shell and tracer
            positions_from_delta = glass.positions_from_delta(
                tracer_dict['N'][i],
                delta_i,
                bias=tracer_dict['bias'],
                rng=rng,
            )
            for lon, lat, count in positions_from_delta:
                # Sample redshifts for these sources
                z = glass.redshifts(count, shells[i], rng=rng)
                # Store sky positions (ra, dec, z)
                sky_pos = np.stack([lon, lat, z], axis=-1)
                print(f"Tracer {tracer_name}, Shell {i}: {len(sky_pos)} positions")
                positions_sky[tracer_name].append(sky_pos)
                # Store 3D positions
                pos = compute_3d_positions(lon, lat, z)
                positions_3d[tracer_name].append(pos)
    
    # Concatenate all positions for each tracer into a single array
    for name in positions_3d:
        positions_3d[name] = np.concatenate(positions_3d[name], axis=0)
    for name in positions_sky:
        positions_sky[name] = np.concatenate(positions_sky[name], axis=0)
        # Normalize the right ascension (RA) to be in the range [0, 360)
        positions_sky[name][:, 0] = positions_sky[name][:, 0] % 360
    
    print(f"Galaxies positions shape: {positions_sky['galaxies'].shape}")
    print(f"AGN positions shape: {positions_sky['agn'].shape}")
    
    # Prepare RA and Dec arrays for plotting from positions_sky
    ra_gal = positions_sky['galaxies'][:, 0]
    ra_agn = positions_sky['agn'][:, 0]
    dec_gal = positions_sky['galaxies'][:, 1]
    dec_agn = positions_sky['agn'][:, 1]
    z_gal = positions_sky['galaxies'][:, 2]
    z_agn = positions_sky['agn'][:, 2]
    
    # Save mock catalog
    if save:
        save_mock_catalog(fn_mock, ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn,
                         len(ra_gal), len(ra_agn), bias_gal, bias_agn,
                         z_max, nside, seed, h, Oc, Ob)
    
    # Return the catalog data
    attrs = {
        'n_gal': len(ra_gal),
        'n_agn': len(ra_agn),
        'b1_gal': bias_gal,
        'b1_agn': bias_agn,
        'z_max': z_max,
        'nside': nside,
        'seed': seed,
        'h': h,
        'Oc': Oc,
        'Ob': Ob
    }
    
    return ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, attrs


def inject_gw_sources(fn_mock, f_agn=0.25, N_gw=1000, gw_seed=None, lambda_agn=0.5,
                      save=True, overwrite_gws=False):
    """Inject GW sources into an existing mock catalog."""
    
    # Load the mock catalog
    print(f"Loading mock catalog from {fn_mock}")
    ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, catalog_attrs = load_mock_catalog(fn_mock)
    
    # Set up random number generator for GW selection
    if gw_seed is None:
        gw_seed = catalog_attrs['seed'] + 1000  # Different seed for GW selection
    rng = np.random.default_rng(seed=gw_seed)
    
    # Generate filename for GW injection
    dir_mock = os.path.dirname(fn_mock)
    tag_gw = f'_fagn{f_agn}_lambdaagn{lambda_agn}_N{N_gw}_seed{gw_seed}'
    fn_gw = os.path.join(dir_mock, f'gws{tag_gw}.h5')
    
    # Check if GW injection already exists
    if os.path.exists(fn_gw) and not overwrite_gws:
        print(f"GW injection already exists: {fn_gw}")
        print("Loading existing GW injection...")
        return load_gw_injection(fn_gw)
    
    if os.path.exists(fn_gw) and overwrite_gws:
        print(f"GW injection exists but overwrite_gws=True, regenerating: {fn_gw}")
    else:
        print("Creating new GW injection...")
    
    print("Selecting GW sources...")
    N_gal = len(ra_gal)
    N_agn = len(ra_agn)
    
    # Calculate fractions
    frac_gal, frac_agn = utils.compute_gw_host_fractions(N_gal, N_agn, f_agn, lambda_agn=lambda_agn)
    
    N_gw_gal = round(frac_gal * N_gw)
    N_gw_agn = round(frac_agn * N_gw)
    
    print(f"Number of GW sources in galaxies: {N_gw_gal}")
    print(f"Number of GW sources in AGN: {N_gw_agn}")
    print(f"Fraction in galaxies: {frac_gal}")
    print(f"Fraction in AGN: {frac_agn}")

    # Randomly select indices for GW sources in galaxies and AGN
    i_gw_gal = rng.choice(np.arange(N_gal), N_gw_gal, replace=False)
    i_gw_agn = rng.choice(np.arange(N_agn), N_gw_agn, replace=False)
    
    # Save GW injection data
    if save:
        # Create output directory if it doesn't exist
        save_gw_injection(fn_gw, i_gw_gal, i_gw_agn, N_gw, f_agn, lambda_agn, gw_seed)
    
    return i_gw_gal, i_gw_agn, N_gw, f_agn, lambda_agn, gw_seed


if __name__ == "__main__":
    config, overwrite_mock, overwrite_gws = parse_args()
    main(config, overwrite_mock=overwrite_mock, overwrite_gws=overwrite_gws)
