import numpy as np 


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def compute_gw_host_fractions(N_gal, N_agn, f_agn, lambda_agn):
    """Compute the fraction of AGN and galaxies hosting GWs
    based on the number of objects and the physical AGN fraction."""

    W_gal = (1-lambda_agn) * N_gal
    W_agn = lambda_agn * N_agn
    W_tot = W_gal + W_agn

    alpha_agn_disk = f_agn
    alpha_agn_field = (1-f_agn) * W_agn/W_tot
    alpha_gal = (1-f_agn) * W_gal/W_tot
    alpha_agn = alpha_agn_disk + alpha_agn_field
    alpha_tot = alpha_agn + alpha_gal

    # print(f"Alpha per object if f=0, lambda=0.5: {1/(N_gal + N_agn):.4e}")
    # print(f"Alpha gal: {alpha_gal:.4f}")
    # print(f"Alpha agn: {alpha_agn:.4f}")
    # print(f"Alpha agn disk: {alpha_agn_disk:.4f}")
    # print(f"Alpha agn non-disk: {alpha_agn_field:.4f}")
    # print(f"Total alpha (should=1): {alpha_tot:.4f}")
    return alpha_gal, alpha_agn