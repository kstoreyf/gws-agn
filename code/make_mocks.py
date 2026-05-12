#!/usr/bin/env python3
"""
Mock galaxy and AGN catalog generation using GLASS.

This script generates mock galaxy and AGN catalogs using the GLASS (Gaussian Lognormal 
Astronomical Simulation Software) package. It creates matter fields, populates them 
with tracers (galaxies and AGN), and saves the results to an HDF5 file.

The script is now separated into two main functions:
1. create_mock_catalog_glass() - Creates the base mock catalog (galaxies and AGN) with GLASS
2. create_mock_catalog_uniform() - Same HDF5 layout: uniform on the sky, z uniform in comoving volume
3. inject_gw_sources() - Injects GW sources into an existing mock catalog

Which catalog builder runs is selected by ``paths.tag_mocktype`` in the YAML config: ``_glass`` (default)
or ``_uniform``. Omit the key for backward compatibility (GLASS).

This allows for multiple GW injection sets from the same base catalog.

Note: GLASS/CAMB import only for GLASS mocks. ``_uniform`` uses Astropy only; ``nbar_*`` are per
square arcminute (same units as GLASS ``ngal``), with Poisson mean ``nbar * ARCMIN2_SPHERE`` (~1.5e6
for ``nbar = 1e-2``). For GLASS you need glass (2025.1), camb, and glass.ext.camb (``pip install glass.ext.camb``).
"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo_astropy
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import h5py
import sys
import os
import argparse
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))
import utils

# Full-sky area in arcmin^2 (same as ``glass.points.ARCMIN2_SPHERE``). GLASS interprets the
# tracer ``nbar`` in configs in the same units as ``ngal`` in ``positions_from_delta``: expected
# count per arcmin^2, so total mean = ``nbar * _ARCMIN2_SPHERE`` (see ``glass.uniform_positions``).
_ARCMIN2_SPHERE = float(60**6 // 100) / np.pi


def _diag(msg):
    """Stdout + flush so logs appear before a possible OOM kill."""
    print(msg, flush=True)


# Heavy GLASS / CAMB imports stay inside ``create_mock_catalog_glass`` only; uniform mocks
# never import GLASS or CAMB.


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
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--overwrite-mock', action='store_true',
                        help='Overwrite existing mock catalog if it exists')
    parser.add_argument('--overwrite-gws', action='store_true',
                        help='Overwrite existing GW injection if it exists')
    args = parser.parse_args()
    
    config_path = args.config

    # Load YAML config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config, args.overwrite_mock, args.overwrite_gws


def main(config, overwrite_mock=False, overwrite_gws=False):
    import time
    t_start = time.perf_counter()
    """Main function to create mock catalog and inject GW sources."""
    
    # Extract parameters from config
    mc = config['mock_catalog']
    seed = mc['seed']
    nbar_gal = mc['nbar_gal']
    nbar_agn = mc['nbar_agn']
    bias_gal = mc['bias_gal']
    bias_agn = mc['bias_agn']
    z_min = mc['z_min']
    z_max = mc['z_max']
    
    # GW injection parameters
    f_agn = config['gw_injection']['f_agn']
    lambda_agn = config['gw_injection']['lambda_agn']
    N_gw = config['gw_injection']['N_gw']
    seed_gw = config['gw_injection']['seed_gw']
    z_max_gw = config['gw_injection'].get('z_max_gw', None)
    
    # Extract cosmology parameters from config
    if 'cosmology' in config:
        h = config['cosmology'].get('h', None)
        H0 = config['cosmology'].get('H0', None)
        Om0 = config['cosmology'].get('Om0', None)
        Ob0 = config['cosmology'].get('Ob0', None)
        
        # Validate h and H0 consistency if both are provided
        if h is not None and H0 is not None:
            expected_H0 = 100.0 * h
            if abs(H0 - expected_H0) > 0.01:  # Allow small floating point differences
                raise ValueError(f"Inconsistent cosmology: h={h} implies H0={expected_H0}, but config has H0={H0}")
        
        # Use h from config if provided, otherwise derive from H0, otherwise use Planck15
        if h is None:
            if H0 is not None:
                h = H0 / 100.0
            else:
                h = cosmo_astropy.h
                print(f"Warning: No cosmology specified in config, using Planck15: h={h}")
        else:
            # If h is provided, ensure H0 matches
            if H0 is None:
                H0 = 100.0 * h
            elif abs(H0 - 100.0 * h) > 0.01:
                raise ValueError(f"Inconsistent cosmology: h={h} and H0={H0} don't match (expected H0={100.0*h})")
        
        # Use Om0 and Ob0 from config if provided, otherwise use Planck15
        if Om0 is None:
            Om0 = cosmo_astropy.Om0
        if Ob0 is None:
            Ob0 = cosmo_astropy.Ob0
    else:
        # No cosmology section in config, use Planck15 defaults
        h = cosmo_astropy.h
        H0 = 100.0 * h
        Om0 = cosmo_astropy.Om0
        Ob0 = cosmo_astropy.Ob0
        print(f"Warning: No cosmology section in config, using Planck15 defaults: h={h}, H0={H0}, Om0={Om0}, Ob0={Ob0}")
    
    # Get paths from config
    dir_mock = config['paths']['dir_mock']
    name_cat = config['paths']['name_cat']
    name_gw = config['paths']['name_gw']
    fn_mock = os.path.join(dir_mock, name_cat)
    fn_gw = os.path.join(dir_mock, name_gw)
    
    tag_mocktype = config.get('paths', {}).get('tag_mocktype', '_glass')
    if tag_mocktype not in ('_glass', '_uniform'):
        raise ValueError(
            f"paths.tag_mocktype must be '_glass' or '_uniform' (default '_glass'), got {tag_mocktype!r}"
        )
    
    _diag(f"main: tag_mocktype={tag_mocktype!r}, fn_mock={fn_mock}")
    print("=== Creating Mock Catalog ===")
    if tag_mocktype == '_glass':
        ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, catalog_attrs = create_mock_catalog_glass(
            seed=seed, nbar_gal=nbar_gal, nbar_agn=nbar_agn,
            bias_gal=bias_gal, bias_agn=bias_agn,
            z_min=z_min, z_max=z_max,
            nside=mc['nside'],
            lmax=mc.get('lmax', mc['nside']),
            fn_mock=fn_mock, save=True, overwrite_mock=overwrite_mock,
            h=h, Om0=Om0, Ob0=Ob0
        )
        _diag("main: create_mock_catalog_glass returned")
    else:
        _diag("main: calling create_mock_catalog_uniform ...")
        ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, catalog_attrs = create_mock_catalog_uniform(
            seed=seed, nbar_gal=nbar_gal, nbar_agn=nbar_agn,
            z_min=z_min, z_max=z_max,
            nside=mc['nside'],
            fn_mock=fn_mock, save=True, overwrite_mock=overwrite_mock,
            h=h, Om0=Om0, Ob0=Ob0
        )
        _diag("main: create_mock_catalog_uniform returned")
    # inject_gw_sources reloads the catalog from disk; drop in-memory copy first to halve peak RAM
    _diag("main: freeing in-memory catalog arrays before GW injection")
    del ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, catalog_attrs
    _diag("main: catalog arrays freed")

    print("\n=== Injecting GW Sources ===")

    i_gw_gal, i_gw_agn = inject_gw_sources(
        fn_mock, fn_gw, f_agn, lambda_agn, N_gw, seed_gw,
        z_max_gw=z_max_gw, save=True, overwrite_gws=overwrite_gws
    )
    
    print("\nMock catalog generation and GW injection complete!")
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    minutes = elapsed / 60
    print(f"Total time: {elapsed:.2f} s = {minutes:.2f} min")


def compute_3d_positions(lon, lat, redshift):
    """Compute 3D Cartesian positions from lon, lat, and redshift arrays."""
    x = redshift * np.cos(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
    y = redshift * np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
    z = redshift * np.sin(np.deg2rad(lat))
    return np.stack([x, y, z], axis=-1)


def save_mock_catalog(fn_mock, ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, attrs):
    """Save mock catalog data to HDF5 file (without GW sources).

    Parameters
    ----------
    attrs : dict
        Scalar entries written to the file root attributes via ``f.attrs``.
    """
    compression = 'gzip'
    compression_opts = 9
    _diag(
        f"save_mock_catalog: opening {fn_mock} for write "
        f"(ra_gal n={len(ra_gal)}, ra_agn n={len(ra_agn)})"
    )
    with h5py.File(fn_mock, 'w') as f:
        _diag("save_mock_catalog: writing ra_gal ...")
        f.create_dataset('ra_gal', data=ra_gal, compression=compression, compression_opts=compression_opts)
        _diag("save_mock_catalog: writing dec_gal ...")
        f.create_dataset('dec_gal', data=dec_gal, compression=compression, compression_opts=compression_opts)
        _diag("save_mock_catalog: writing z_gal ...")
        f.create_dataset('z_gal', data=z_gal, compression=compression, compression_opts=compression_opts)
        _diag("save_mock_catalog: writing ra_agn ...")
        f.create_dataset('ra_agn', data=ra_agn, compression=compression, compression_opts=compression_opts)
        _diag("save_mock_catalog: writing dec_agn ...")
        f.create_dataset('dec_agn', data=dec_agn, compression=compression, compression_opts=compression_opts)
        _diag("save_mock_catalog: writing z_agn ...")
        f.create_dataset('z_agn', data=z_agn, compression=compression, compression_opts=compression_opts)
        _diag("save_mock_catalog: writing file attrs ...")
        for key, val in attrs.items():
            f.attrs[key] = val
    
    print(f"Mock catalog saved to {fn_mock}")


def load_mock_catalog(fn_mock):
    """Load mock catalog from HDF5 file."""
    _diag(f"load_mock_catalog: opening {fn_mock} for read")
    with h5py.File(fn_mock, 'r') as f:
        # Load position data
        _diag("load_mock_catalog: reading ra_gal ...")
        ra_gal = f['ra_gal'][:]
        _diag("load_mock_catalog: reading dec_gal ...")
        dec_gal = f['dec_gal'][:]
        _diag("load_mock_catalog: reading z_gal ...")
        z_gal = f['z_gal'][:]
        _diag("load_mock_catalog: reading ra_agn ...")
        ra_agn = f['ra_agn'][:]
        _diag("load_mock_catalog: reading dec_agn ...")
        dec_agn = f['dec_agn'][:]
        _diag("load_mock_catalog: reading z_agn ...")
        z_agn = f['z_agn'][:]
        
        # Load attributes
        _diag("load_mock_catalog: copying attrs ...")
        attrs = dict(f.attrs)
    _bytes = (
        ra_gal.nbytes
        + dec_gal.nbytes
        + z_gal.nbytes
        + ra_agn.nbytes
        + dec_agn.nbytes
        + z_agn.nbytes
    )
    _diag(
        f"load_mock_catalog: done (n_gal={len(ra_gal)}, n_agn={len(ra_agn)}, "
        f"~{_bytes / 1e9:.3f} GB raw float64 for six position arrays)"
    )
    return ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, attrs


def save_gw_injection(fn_gw, i_gw_gal, i_gw_agn, N_gw, f_agn, lambda_agn, seed_gw, z_max_gw=None):
    """Save GW injection data to HDF5 file."""
    compression = 'gzip'
    compression_opts = 9
    _diag(
        f"save_gw_injection: opening {fn_gw} (len i_gw_gal={len(i_gw_gal)}, len i_gw_agn={len(i_gw_agn)})"
    )
    with h5py.File(fn_gw, 'w') as f:
        # Create datasets with compression
        _diag("save_gw_injection: writing i_gw_gal ...")
        f.create_dataset('i_gw_gal', data=i_gw_gal, compression=compression, compression_opts=compression_opts)
        _diag("save_gw_injection: writing i_gw_agn ...")
        f.create_dataset('i_gw_agn', data=i_gw_agn, compression=compression, compression_opts=compression_opts)
        
        # attributes
        _diag("save_gw_injection: writing attrs ...")
        f.attrs['n_gw'] = N_gw
        f.attrs['f_agn'] = f_agn
        f.attrs['lambda_agn'] = lambda_agn
        f.attrs['seed_gw'] = seed_gw
        if z_max_gw is not None:
            f.attrs['z_max_gw'] = z_max_gw
    
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


def create_mock_catalog_glass(seed, nbar_gal, nbar_agn, bias_gal, bias_agn,
                       z_min=0.0, z_max=1.5, nside=128, lmax=128, fn_mock=None, save=True, overwrite_mock=False,
                       h=None, Om0=None, Ob0=None):
    """
    Create a mock galaxy and AGN catalog.
    
    Parameters
    ----------
    seed : int
        Random seed for catalog generation
    nbar_gal : float
        Number density of galaxies
    nbar_agn : float
        Number density of AGN
    bias_gal : float
        Bias parameter for galaxies
    bias_agn : float
        Bias parameter for AGN
    z_min : float
        Minimum redshift (default: 0.0)
    z_max : float
        Maximum redshift (default: 1.5)
    nside : int
        HEALPix resolution parameter (default: 128)
    lmax : int
        Maximum multipole (default: 128)
    fn_mock : str
        Output filename for mock catalog
    save : bool
        Whether to save the catalog (default: True)
    overwrite_mock : bool
        Whether to overwrite existing catalog (default: False)
    h : float, optional
        Dimensionless Hubble parameter (default: None, uses Planck15)
    Om0 : float, optional
        Matter density parameter (default: None, uses Planck15)
    Ob0 : float, optional
        Baryon density parameter (default: None, uses Planck15)
    """
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
    
    _diag("glass: importing camb, glass, glass.ext.camb ...")
    import camb
    from cosmology import Cosmology
    import glass
    import glass.ext.camb
    _diag("glass: heavy imports finished")
    
    # Creating a numpy random number generator for sampling
    rng = np.random.default_rng(seed=seed)
    
    # Cosmology for the simulation - use provided values or fall back to Planck15
    if h is None:
        h = cosmo_astropy.h
    if Om0 is None:
        Om0 = cosmo_astropy.Om0
    if Ob0 is None:
        Ob0 = cosmo_astropy.Ob0
    
    Oc = Om0 - Ob0
    print(f"Using cosmology: h = {h}, H0 = {100.0*h}, Om0 = {Om0}, Oc = {Oc}, Ob = {Ob0}")
    
    # Set up CAMB parameters for matter angular power spectrum
    pars = camb.set_params(
        H0=100 * h,
        omch2=Oc * h**2,
        ombh2=Ob0 * h**2,
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
    
    _diag("glass: materializing matter generator to list (peak RAM) ...")
    matter = list(matter)  # Convert generator to a list for multiple iterations
    _diag(f"glass: matter list built, n_shells={len(matter)}")
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
    _diag("glass: concatenating 3d positions ...")
    for name in positions_3d:
        positions_3d[name] = np.concatenate(positions_3d[name], axis=0)
    _diag("glass: concatenating sky positions ...")
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
        'Ob': Ob0,
    }
    if save:
        save_mock_catalog(fn_mock, ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, attrs)
    
    return ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, attrs


def create_mock_catalog_uniform(
    seed, nbar_gal, nbar_agn,
    z_min=0.0, z_max=1.5, nside=128, fn_mock=None, save=True, overwrite_mock=False,
    h=None, Om0=None, Ob0=None,
):
    """
    Create a mock galaxy and AGN catalog uniform on the sphere with redshifts drawn
    uniformly in comoving volume between z_min and z_max (constant comoving number density).

    Uses the same HDF5 layout and ``nbar_*`` keys as GLASS mocks. Those numbers are treated in
    the same units as GLASS ``ngal`` in ``positions_from_delta`` / ``uniform_positions``: expected
    sources **per square arcminute** on the sphere, so ``E[N] = nbar * _ARCMIN2_SPHERE`` (order
    ~10^6 for ``nbar = 1e-2``). This is not ``nbar * comoving_volume / h**3``. Redshifts are
    uniform in comoving volume; sky positions are isotropic. Bias attrs are not written.
    """
    _diag(
        f"uniform: enter create_mock_catalog_uniform "
        f"(z_min={z_min}, z_max={z_max}, nside={nside}, save={save}, fn_mock={fn_mock})"
    )
    dir_mock = os.path.dirname(fn_mock)
    _diag(f"uniform: makedirs {dir_mock!r}")
    os.makedirs(dir_mock, exist_ok=True)
    
    if os.path.exists(fn_mock) and not overwrite_mock:
        print(f"Mock catalog already exists: {fn_mock}")
        print("Loading existing catalog...")
        return load_mock_catalog(fn_mock)
    
    if os.path.exists(fn_mock) and overwrite_mock:
        print(f"Mock catalog exists but overwrite_mock=True, regenerating: {fn_mock}")
    else:
        print("Creating new uniform-sky mock catalog...")
    
    _diag("uniform: creating RNG")
    rng = np.random.default_rng(seed=seed)
    
    if h is None:
        h = cosmo_astropy.h
    if Om0 is None:
        Om0 = cosmo_astropy.Om0
    if Ob0 is None:
        Ob0 = cosmo_astropy.Ob0
    
    Oc = Om0 - Ob0
    print(f"Using cosmology (uniform mock): h = {h}, H0 = {100.0*h}, Om0 = {Om0}, Oc = {Oc}, Ob = {Ob0}")
    
    _diag("uniform: building FlatLambdaCDM ...")
    cosmo = FlatLambdaCDM(H0=100.0 * h, Om0=Om0, Ob0=Ob0, Tcmb0=2.7255)
    _diag("uniform: FlatLambdaCDM ready")
    
    _diag("uniform: allocating z_grid (8192) and comoving_volume ...")
    z_grid = np.linspace(z_min, z_max, 8192)
    _diag(f"uniform: z_grid shape={z_grid.shape}, nbytes={z_grid.nbytes}")
    V = cosmo.comoving_volume(z_grid).to_value(u.Mpc**3)
    _diag(f"uniform: V array nbytes={V.nbytes}")
    V_rel = V - V[0]
    delta_V = V_rel[-1]
    _diag(f"uniform: V_rel nbytes={V_rel.nbytes}, delta_V={delta_V:.6e} Mpc^3 (for z sampling only)")
    
    _diag("uniform: mean counts = nbar * _ARCMIN2_SPHERE (per-arcmin^2 convention, no GLASS import) ...")
    mean_gal = float(nbar_gal) * _ARCMIN2_SPHERE
    mean_agn = float(nbar_agn) * _ARCMIN2_SPHERE
    N_gal = int(rng.poisson(mean_gal))
    N_agn = int(rng.poisson(mean_agn))
    est_pos_bytes = 8 * 3 * (N_gal + N_agn)  # six float64 columns total (gal + agn)
    _diag(
        f"uniform: Poisson counts drawn — "
        f"E[N_gal]={mean_gal:.2f} -> N_gal={N_gal}, E[N_agn]={mean_agn:.2f} -> N_agn={N_agn}; "
        f"~{est_pos_bytes / 1e9:.3f} GB if six full float64 position arrays materialized"
    )
    print(
        f"Uniform mock (nbar as per-arcmin^2): E[N_gal]={mean_gal:.2f} -> N_gal={N_gal}, "
        f"E[N_agn]={mean_agn:.2f} -> N_agn={N_agn}; "
        f"Delta_V={delta_V:.4e} Mpc^3 used only for z ~ volume"
    )
    
    def sample_z_uniform_comoving_volume(n):
        if n == 0:
            return np.array([], dtype=float)
        _diag(f"uniform: sample_z_uniform_comoving_volume(n={n}): drawing u, interp z ...")
        u = rng.random(n) * V_rel[-1]
        z_out = np.interp(u, V_rel, z_grid)
        _diag(f"uniform: sample_z_uniform_comoving_volume done, z_out nbytes={z_out.nbytes}")
        return z_out
    
    def sample_uniform_sphere(n):
        if n == 0:
            return np.empty(0), np.empty(0)
        _diag(f"uniform: sample_uniform_sphere(n={n}): drawing ra/dec ...")
        ra = rng.uniform(0.0, 360.0, n)
        sin_dec = rng.uniform(-1.0, 1.0, n)
        dec = np.degrees(np.arcsin(sin_dec))
        _diag(
            f"uniform: sample_uniform_sphere done, ra nbytes={ra.nbytes}, dec nbytes={dec.nbytes}"
        )
        return ra, dec
    
    _diag("uniform: sampling z_gal ...")
    z_gal = sample_z_uniform_comoving_volume(N_gal)
    _diag("uniform: sampling z_agn ...")
    z_agn = sample_z_uniform_comoving_volume(N_agn)
    _diag("uniform: sampling ra_gal, dec_gal ...")
    ra_gal, dec_gal = sample_uniform_sphere(N_gal)
    _diag("uniform: sampling ra_agn, dec_agn ...")
    ra_agn, dec_agn = sample_uniform_sphere(N_agn)
    
    _diag(
        f"uniform: all sky+z samples done — "
        f"nbytes z_gal={z_gal.nbytes}, z_agn={z_agn.nbytes}, "
        f"ra_gal={ra_gal.nbytes}, dec_gal={dec_gal.nbytes}, ra_agn={ra_agn.nbytes}, dec_agn={dec_agn.nbytes}"
    )
    
    attrs = {
        'n_gal': len(ra_gal),
        'n_agn': len(ra_agn),
        'z_max': z_max,
        'nside': nside,
        'seed': seed,
        'h': h,
        'Oc': Oc,
        'Ob': Ob0,
    }
    _diag(f"uniform: attrs built n_gal={attrs['n_gal']}, n_agn={attrs['n_agn']}")
    if save:
        _diag("uniform: calling save_mock_catalog ...")
        save_mock_catalog(fn_mock, ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, attrs)
        _diag("uniform: save_mock_catalog returned")
    else:
        _diag("uniform: save=False, skipping write")
    
    _diag("uniform: create_mock_catalog_uniform returning")
    return ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, attrs


def inject_gw_sources(fn_mock, fn_gw, f_agn, lambda_agn, N_gw, seed_gw,
                      z_max_gw=None, save=True, overwrite_gws=False):
    """Inject GW sources into an existing mock catalog."""
    _diag(
        f"inject_gw_sources: start fn_mock={fn_mock}, fn_gw={fn_gw}, "
        f"N_gw={N_gw}, z_max_gw={z_max_gw}, save={save}"
    )
    
    # Load the mock catalog
    print(f"Loading mock catalog from {fn_mock}")
    ra_gal, dec_gal, z_gal, ra_agn, dec_agn, z_agn, catalog_attrs = load_mock_catalog(fn_mock)
    _diag("inject_gw_sources: load_mock_catalog returned")
    
    # Set up random number generator for GW selection
    rng = np.random.default_rng(seed=seed_gw)
    
    # Check if GW injection already exists
    if os.path.exists(fn_gw) and not overwrite_gws:
        print(f"GW injection already exists: {fn_gw}")
        print("Loading existing GW injection...")
        i_gw_gal, i_gw_agn, _ = load_gw_injection(fn_gw)
        return i_gw_gal, i_gw_agn
    
    if os.path.exists(fn_gw) and overwrite_gws:
        print(f"GW injection exists but overwrite_gws=True, regenerating: {fn_gw}")
    else:
        print("Creating new GW injection...")
    
    print("Selecting GW sources...")

    # Restrict eligible host pool to z <= z_max_gw if specified
    _diag(
        f"inject_gw_sources: allocating eligible index aranges "
        f"(n_z_gal={len(z_gal)}, n_z_agn={len(z_agn)}) ..."
    )
    eligible_gal = np.arange(len(z_gal))
    eligible_agn = np.arange(len(z_agn))
    _diag(
        f"inject_gw_sources: eligible aranges nbytes "
        f"gal={eligible_gal.nbytes}, agn={eligible_agn.nbytes}"
    )
    if z_max_gw is not None:
        eligible_gal = eligible_gal[z_gal <= z_max_gw]
        eligible_agn = eligible_agn[z_agn <= z_max_gw]
        print(f"z_max_gw={z_max_gw}: eligible hosts: {len(eligible_gal)} galaxies, {len(eligible_agn)} AGN "
              f"(out of {len(z_gal)} gal, {len(z_agn)} AGN total)")

    N_gal = len(eligible_gal)
    N_agn = len(eligible_agn)
    
    # Calculate fractions
    frac_gal, frac_agn = utils.compute_gw_host_fractions(N_gal, N_agn, f_agn, lambda_agn)
    
    N_gw_gal = round(frac_gal * N_gw)
    N_gw_agn = N_gw - N_gw_gal
    
    print(f"Number of GW sources in galaxies: {N_gw_gal}")
    print(f"Number of GW sources in AGN: {N_gw_agn}")
    print(f"Fraction in galaxies: {frac_gal}")
    print(f"Fraction in AGN: {frac_agn}")

    # Randomly select from eligible hosts
    _diag(f"inject_gw_sources: rng.choice gal pool (N_gw_gal={N_gw_gal}) ...")
    i_gw_gal = rng.choice(eligible_gal, N_gw_gal, replace=False)
    _diag(f"inject_gw_sources: rng.choice agn pool (N_gw_agn={N_gw_agn}) ...")
    i_gw_agn = rng.choice(eligible_agn, N_gw_agn, replace=False)
    _diag(
        f"inject_gw_sources: choices done, i_gw_gal nbytes={i_gw_gal.nbytes}, "
        f"i_gw_agn nbytes={i_gw_agn.nbytes}"
    )

    _diag("inject_gw_sources: building z_all_gw summary array ...")
    z_all_gw = np.concatenate([z_gal[i_gw_gal], z_agn[i_gw_agn]]) if N_gw_agn > 0 else z_gal[i_gw_gal]
    print(f"Injected GW host redshifts: z_min={z_all_gw.min():.4f}, z_max={z_all_gw.max():.4f} "
          f"(catalog z_max={catalog_attrs['z_max']})")
    
    # Save GW injection data
    if save:
        _diag("inject_gw_sources: calling save_gw_injection ...")
        save_gw_injection(fn_gw, i_gw_gal, i_gw_agn, N_gw, f_agn, lambda_agn, seed_gw, z_max_gw=z_max_gw)
        _diag("inject_gw_sources: save_gw_injection returned")
    
    _diag("inject_gw_sources: returning")
    return i_gw_gal, i_gw_agn


if __name__ == "__main__":
    _diag("make_mocks: __main__ starting (parse_args) ...")
    config, overwrite_mock, overwrite_gws = parse_args()
    _diag("make_mocks: parse_args done, entering main ...")
    main(config, overwrite_mock=overwrite_mock, overwrite_gws=overwrite_gws)
