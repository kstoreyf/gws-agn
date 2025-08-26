import jax
import jax.numpy as jnp
from jax import jit
from scipy.interpolate import interp2d as scipy_interp2d
import numpy as np

class CartesianGrid:
    """Simple CartesianGrid class for compatibility"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

@jit
def interp2d(x, y, x_grid, y_grid, z_grid):
    """
    Simple 2D interpolation using JAX
    
    Parameters:
    -----------
    x, y : arrays
        Points to interpolate at
    x_grid, y_grid : arrays
        Grid coordinates
    z_grid : array
        Grid values to interpolate from
        
    Returns:
    --------
    Interpolated values at (x, y)
    """
    # Convert to numpy for scipy interpolation
    x_np = np.array(x)
    y_np = np.array(y)
    x_grid_np = np.array(x_grid)
    y_grid_np = np.array(y_grid)
    z_grid_np = np.array(z_grid)
    
    # Create scipy interpolator
    interp_func = scipy_interp2d(x_grid_np, y_grid_np, z_grid_np, kind='linear')
    
    # Interpolate
    result = interp_func(x_np, y_np)
    
    # Handle single value case
    if np.isscalar(x_np):
        result = result.item()
    
    return jnp.array(result) 