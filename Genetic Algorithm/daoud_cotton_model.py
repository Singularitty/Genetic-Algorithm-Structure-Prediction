import numpy as np
import settings

f = settings.f
core_radius = settings.core_radius

def potential_energy(r):
    """
    Takes a float representing the distance between two particles.
    Returns a float representing the Daoud-Cotton model potential energy of that pair of particles.
    """
    if r <= core_radius:
        return (5/18)*(f**(1.5)) * (-np.log(r/core_radius) + 1.0 / (1.0 + np.sqrt(f/2)))
    else:
        return ((5/18)*(f**(1.5))) * (core_radius / (1 + np.sqrt(f/2)) * (np.exp((np.sqrt(f)*(r - core_radius))/(2*core_radius)) / r))