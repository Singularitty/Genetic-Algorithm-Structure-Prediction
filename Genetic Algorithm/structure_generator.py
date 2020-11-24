import numpy as np
import settings

a = settings.a

# Definition of lattice vectors

def triangular_lat(ind):    # Triangular latice vectors
    """
    Takes a list containg the genes that make up an individual.
    Returns a list with two tuples, were each tuple represents a primitive vector of a triangular lattice.
    """
    return [ (a, 0.0) , (a*ind[0]*0.5, a*ind[0]*(np.sqrt(3)/2)) ]

def x(ind):                 # General lattice vectors (depend on the individual)
    """
    Takes a list containg the genes that make up an individual.
    Returns a list with two tuples, were each tuple represents a primitive vector of a lattice that depends on the individuals genes.
    """
    return [ (a, 0.0) , (a*ind[0]*np.cos(ind[1]), a*ind[0]*np.sin(ind[1])) ]

# Conversao de todas as estructuras equivalentes para uma estructura unica com a menor circunferencia

def unique_lattice(lattice_vectors,ind):
    """
    Takes a list with two tuples, each representing a lattice vector and a list with the genes of an individual.
    Returns a list with two tuples, representing the equivalent lattice vectors with the smallest cell circunference.
    """
    x_1 = lattice_vectors(0,ind)
    x_2 = lattice_vectors(1,ind)
    lattices = [[(x_1[0]+x_2[0] if (x_1[0]+x_2[0]) > 0 else (x_1[0]-x_2[0]),   x_1[1]+x_2[1] if (x_1[1]+x_2[1]) > 0 else x_1[1]-x_2[1])   ,x_2],
                [(x_1[0]-x_2[0] if (x_1[0]-x_2[0]) > 0 else x_1[0]+x_2[0],   x_1[1]-x_2[1] if (x_1[1]-x_2[1]) > 0 else x_1[1]+x_2[1])   ,x_2],
                [x_1,    (x_1[0]+x_2[0] if (x_1[0]+x_2[0]) > 0 else x_1[0]-x_2[0],   x_1[1]+x_2[1] if (x_1[1]+x_2[1]) > 0 else x_1[1]-x_2[1])],
                [x_1,    (x_1[0]-x_2[0] if (x_1[0]-x_2[0]) > 0 else x_1[0]+x_2[0],    x_1[1]-x_2[1] if (x_1[1]-x_2[1]) > 0 else x_1[1]+x_2[1])]]
    
    lattice_radius  = []
    
    for lat in lattices:
        point_1 = lat[0]
        point_2 = lat[1]
        m_a = (point_2[1]-point_1[1])/(point_2[0]-point_1[0])
        m_b = point_2[1]/point_2[0]
        x = (m_a*m_b*(point_1[1]) + m_b*(point_1[0]+point_2[0]) - m_a*(point_2[0])) / 2*(m_b-m_a)
        y = (-1 / m_a) * (x - (point_1[0]-point_2[1])/2) + (point_1[1]-point_2[1])/2
        radius_1 = np.sqrt((x-point_1[0])**2 + (y-point_1[1])**2)
        radius_2 = np.sqrt((x-point_2[0])**2 + (y-point_2[1])**2)
        if radius_1 >= radius_2:
            lattice_radius.append(radius_1)
        else:
            lattice_radius.append(radius_2)
    
    return lattices[lattice_radius.index(min(lattice_radius))]

def structure(lattice_vectors,ind):
    """
    Takes a list with two tuples where each tuple is a lattice vector, and it takes a list containing the genes of an individual.
    Returns a list with tuples, where the tuples represent the positions of particles in a structure made by up of the original lattice structure multiplied in both directons of space.
    """
    
    n = 5
    
    vec_1 = lattice_vectors[0]
    vec_2 = lattice_vectors[1]
    
    c21 = ind[2]
    c22 = ind[3]
    c31 = ind[4]
    c32 = ind[5]
    c41 = ind[6]
    c42 = ind[7]
    
    particle_positions = []

    primitive_vectors = [(c21*vec_1[0]+c22*vec_2[0],c21*vec_1[1]+c22*vec_2[1]),
                         (c31*vec_1[0]+c32*vec_2[0],c31*vec_1[1]+c32*vec_2[1]),
                         (c41*vec_1[0]+c42*vec_2[0],c41*vec_1[1]+c42*vec_2[1])]

    for i in range(0,n+1):
        for j in range(0,n+1):
            for k in range(0,n+1):
                pos = (i*primitive_vectors[0][0]+j*primitive_vectors[1][0]+k*primitive_vectors[2][0],
                       i*primitive_vectors[0][1]+j*primitive_vectors[1][1]+k*primitive_vectors[2][1])
                if (pos not in particle_positions):
                    particle_positions.append(pos)


    return particle_positions
