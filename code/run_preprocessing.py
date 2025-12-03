#!/usr/bin/env python3
"""
Preprocessing script for lognormal catalogs
Converts the catalog into a pixelated format for inference
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax
from jax import random, jit, vmap, grad
from jax import numpy as jnp
from jax.lax import cond

import astropy
import numpy as np
import healpy as hp
import h5py
import astropy.units as u

from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
import astropy.constants as constants
from jax.scipy.special import logsumexp
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from tqdm import tqdm


jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'highest')

# Import jaxinterp2d
from jaxinterp2d import interp2d, CartesianGrid

def main():
    print("Starting preprocessing...")
    
    # Load the catalog 
    filepath = '../data/mocks_glass/mock_seed42_ratioNgalNagn1_bgal1.0_bagn1.0/'
    #filepath = '../data/mocks_glass/mock_seed42_ratioNgalNagn100_bgal0.0_bagn0.0/'
    file = filepath + 'mock_catalog.h5'
    print(f"Loading catalog from {filepath}")
    
    with h5py.File(file, 'r') as f:
        ras_ = np.asarray(f['ra_gal'])*np.pi/180
        decs_ = np.asarray(f['dec_gal'])*np.pi/180
        zs_ = np.asarray(f['z_gal'])
        
    ngals = len(ras_)
    print(f"Loaded {ngals} galaxies")
    
    # Process galaxies
    print("Processing galaxies...")
    nside = 64
    npix = hp.pixelfunc.nside2npix(nside)
    apix = hp.pixelfunc.nside2pixarea(nside)
    pixgrid = np.arange(npix)
    print(f"Number of pixels: {pixgrid.shape}")
    
    ind = hp.pixelfunc.ang2pix(nside, np.pi/2-decs_, ras_)
    
    # Process galaxies without multiprocessing to avoid JAX issues
    cats = []
    ngalaxies = []
    
    maxgals = 350
    for pix in tqdm(pixgrid, desc="Processing galaxies"):
        id_pix = np.where(ind == pix)[0]
        gals = zs_[id_pix]
        ngals_pix = gals.shape[0]
        zgals = [gals]
        
        if ngals_pix < maxgals:
            length = int(maxgals - ngals_pix)
            zgals.append(100*np.ones(length))
        
        cats.append(np.concatenate(zgals))
        ngalaxies.append(ngals_pix)
    
    # Save galaxy data
    print("Saving galaxy data...")
    m = np.asarray(ngalaxies)
    print(f"Max galaxies per pixel: {m.max()}")
    
    with h5py.File(f'{filepath}lognormal_pixelated_nside_{nside}_galaxies.h5', 'w') as f:
        f.attrs['nside'] = nside
        f.create_dataset('zgals', data=np.asarray(cats), compression='gzip', shuffle=False)
        f.create_dataset('ngals', data=m, compression='gzip', shuffle=False)
    
    print("Galaxy preprocessing complete!")
    
    # Process AGN
    print("Processing AGN...")
    with h5py.File(file, 'r') as f:
        ras_ = np.asarray(f['ra_agn'])*np.pi/180
        decs_ = np.asarray(f['dec_agn'])*np.pi/180
        zs_ = np.asarray(f['z_agn'])
        
    nagn = len(ras_)
    print(f"Loaded {nagn} AGN")
    
    ind = hp.pixelfunc.ang2pix(nside, np.pi/2-decs_, ras_)
    
    cats_agn = []
    nagn_pix = []
    
    for pix in tqdm(pixgrid, desc="Processing AGN"):
        id_pix = np.where(ind == pix)[0]
        agn = zs_[id_pix]
        nagn_pix_count = agn.shape[0]
        zagn = [agn]
        
        if nagn_pix_count < maxgals:
            length = int(maxgals - nagn_pix_count)
            zagn.append(100*np.ones(length))
        
        cats_agn.append(np.concatenate(zagn))
        nagn_pix.append(nagn_pix_count)
    
    # Save AGN data
    print("Saving AGN data...")
    m_agn = np.asarray(nagn_pix)
    print(f"Max AGN per pixel: {m_agn.max()}")
    
    with h5py.File(f'{filepath}lognormal_pixelated_nside_{nside}_agn.h5', 'w') as f:
        f.attrs['nside'] = nside
        f.create_dataset('zagn', data=np.asarray(cats_agn), compression='gzip', shuffle=False)
        f.create_dataset('nagn', data=m_agn, compression='gzip', shuffle=False)
    
    print("AGN preprocessing complete!")
    print("Preprocessing finished successfully!")

if __name__ == "__main__":
    main() 