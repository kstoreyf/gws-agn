import numpy as np 


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def compute_gw_host_fractions(N_gal, N_agn, f_agn):
    """Compute the fraction of AGN and galaxies hosting GWs
    based on the number of objects and the physical AGN fraction."""

    frac_agn_phys = f_agn
    frac_agn_nonphys = (1-f_agn) * N_agn/(N_gal + N_agn)
    frac_gal = (1-f_agn) * N_gal/(N_gal + N_agn)
    frac_agn = frac_agn_phys + frac_agn_nonphys
    frac_tot = frac_agn + frac_gal

    # A_norm = 1.0/(N_agn + (1-f_agn)*(N_gal))
    # frac_agn_phys = A_norm*f_agn*N_agn
    # frac_agn_nonphys = A_norm*(1-f_agn)*N_agn
    # frac_agn = frac_agn_phys + frac_agn_nonphys
    # frac_gal = A_norm*(1-f_agn)*N_gal
    # frac_tot = frac_agn_phys+frac_gal+frac_agn_nonphys

    print(f"Frac per object if f=0: {1/(N_gal + N_agn):.4e}")
    print(f"Frac gal: {frac_gal:.4f}")
    print(f"Frac agn: {frac_agn:.4f}")
    print(f"Frac agn phys: {frac_agn_phys:.4f}")
    print(f"Frac agn nonphys: {frac_agn_nonphys:.4f}")
    print(f"Total fraction (should=1): {frac_tot:.4f}")
    return frac_gal, frac_agn