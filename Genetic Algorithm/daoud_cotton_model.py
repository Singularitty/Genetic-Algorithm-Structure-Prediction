import numpy as np
import settings

f = settings.f
core_radius = settings.core_radius


def potential_energy(distance):
    """
    Takes a float representing the distance between two particles.
    Returns a float representing the Daoud-Cotton model potential energy of that pair of particles.
    """
    constant = ((5 / 18) * (f ** 1.5))
    if distance <= core_radius:
        return constant * (-np.log(distance / core_radius) + 1.0 / (1.0 + np.sqrt(f / 2)))
    else:
        return constant * (core_radius / (1 + np.sqrt(f / 2))
                           * (np.exp((np.sqrt(f) * (distance - core_radius)) / (2 * core_radius)) / distance))
