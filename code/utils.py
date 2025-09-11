import numpy as np 


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def compute_gw_host_fractions(N_gal, N_agn, f_agn, lambda_agn=1):
    """Compute the fraction of AGN and galaxies hosting GWs
    based on the number of objects and the physical AGN fraction."""

    frac_agn_disk = f_agn

    # f_agn_field = (1-f_agn)/2 * lambda_agn
    # f_agn_gal = (1-f_agn)/2 * (1-lambda_agn)
    
    # frac_agn_field = (1-f_agn)/(1+lambda_agn) * lambda_agn * N_agn/(N_gal + N_agn)
    # frac_gal = (1-f_agn)/(1+lambda_agn) * (1-lambda_agn) * N_gal/(N_gal + N_agn)
    # frac_agn = frac_agn_disk + frac_agn_field
    # frac_tot = frac_agn + frac_gal

    # frac_agn_nonphys = (1-f_agn)/(1+lambda_agn) * lambda_agn * N_agn/(N_gal + N_agn)
    # frac_gal = (1-f_agn)/(1+lambda_agn) * (1-lambda_agn) * N_gal/(N_gal + N_agn)
    # frac_agn = frac_agn_phys + frac_agn_nonphys
    # frac_tot = frac_agn + frac_gal

    # frac_agn_disk = f_agn
    # frac_agn_field = (1-f_agn) * N_agn/(N_gal + N_agn)
    # frac_gal = (1-f_agn) * N_gal/(N_gal + N_agn)
    # frac_agn = frac_agn_disk + frac_agn_field
    # frac_tot = frac_agn + frac_gal

    W_gal = (1-lambda_agn) * N_gal
    W_agn = lambda_agn * N_agn
    W_tot = W_gal + W_agn

    frac_agn_disk = f_agn
    frac_agn_field = (1-f_agn) * W_agn/W_tot
    frac_gal = (1-f_agn) * W_gal/W_tot
    frac_agn = frac_agn_disk + frac_agn_field
    frac_tot = frac_agn + frac_gal

    print(f"Frac per object if f=0: {1/(N_gal + N_agn):.4e}")
    print(f"Frac gal: {frac_gal:.4f}")
    print(f"Frac agn: {frac_agn:.4f}")
    print(f"Frac agn disk: {frac_agn_disk:.4f}")
    print(f"Frac agn field: {frac_agn_field:.4f}")
    print(f"Total fraction (should=1): {frac_tot:.4f}")
    return frac_gal, frac_agn