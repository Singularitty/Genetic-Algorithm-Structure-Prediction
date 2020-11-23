import numpy as np
import settings


f = settings.f
core_radius = settings.core_radius
r_cut = settings.r_cut

def r(position_1,position_2):
    """
    Takes two tuples representin the positions of two particles.
    Returns a float represeting the distance in space of those two particles.
    """
    return np.sqrt((position_1[0]-position_2[0])**2 + (position_1[1]-position_2[1])**2)

def daoud_cotton_u(r):
    """
    Takes a float representing the distance between two particles.
    Returns a float representing the Daoud-Cotton model potential energy of that pair of particles.
    """
    if r <= core_radius:
        return (5/18)*(f**(1.5)) * (-np.log(r/core_radius) + 1.0 / (1.0 + np.sqrt(f/2)))
    else:
        return ((5/18)*(f**(1.5))) * (core_radius / (1 + np.sqrt(f/2)) * (np.exp((np.sqrt(f)*(r - core_radius))/(2*core_radius)) / r))

# Ground State Calculation


def avg_potential_energy(position_list):
    """
    Takes a list with tuples which represent the positions of particles in space.
    Returns a float representing the average potential energy for a structure made of those particles for a given potential.
    """
    energy_sum = 0
    n = len(position_list)
    for i in range(1, n):
        for j in range(i+1, n):
            distance = r(position_list[i],position_list[j])
            if distance < r_cut:
                energy_sum += daoud_cotton_u(distance)
    return (0.5*n)*energy_sum