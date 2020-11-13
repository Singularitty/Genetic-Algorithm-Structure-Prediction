import random as rng
from datetime import datetime
import numpy as np


# Daoud-Cotton model

global f            # Number of arms in star polymer
f = 50.0

global core_radius  #Core radius
core_radius = 1

global r_cut        #Cutoff radius
r_cut = 2

global a            #Packing fracture
a = 2.4

def daoud_cotton_u(r):
    """
    Takes a float representing the distance between two particles.
    Returns a float representing the Daoud-Cotton model potential energy of that pair of particles.
    """
    if r <= core_radius:
        return (5/18)*(f**(1.5)) * (-np.log(r/core_radius) + 1.0 / (1.0 + np.sqrt(f/2)))
    else:
        return ((5/18)*(f**(1.5))) * (core_radius / (1 + np.sqrt(f/2)) * (np.exp((np.sqrt(f)*(r - core_radius))/(2*core_radius)) / r))

# Define lattice vectors

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
    
    n = 10
    
    vec_1 = lattice_vectors[0]
    vec_2 = lattice_vectors[1]
    
    c21 = ind[2]
    c22 = ind[3]
    c31 = ind[4]
    c32 = ind[5]
    c41 = ind[6]
    c42 = ind[7]
    
    particle_positions = [(0,0)]
    
    for i in range(1,n+1):
        for j in range(1,n+1):
            pos_2 = (i*(c21*vec_1[0]+c22*vec_2[0]),i*(c21*vec_1[1]+c22*vec_2[1]))
            pos_3 = (i*(c31*vec_1[0]+c32*vec_2[0]),j*(c31*vec_1[1]+c32*vec_2[1]))
            pos_4 = (i*(c41*vec_1[0]+c42*vec_2[0]),j*(c41*vec_1[1]+c42*vec_2[1]))
            particle_positions.append(pos_2)
            particle_positions.append(pos_3)
            particle_positions.append(pos_4)
            
            
    return particle_positions

def r(position_1,position_2):
    """
    Takes two tuples representin the positions of two particles.
    Returns a float represeting the distance in space of those two particles.
    """
    return np.sqrt((position_1[0]-position_2[0])**2 + (position_1[1]-position_2[1])**2)

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


# Fitness function


def e(i):
    """
    Auxiliary function for the fitness function.
    Takes an int representing the current generation of individuals.
    Returns a float.
    """
    return 1.0 + i * (np.log(i) / 40.0)

def Fitness(position_list,position_list_triangular,i):
    """
    Takes a list with tuples representing the positions of particles, takes a int representing the current generation of individuals.
    Returns a float, representing the Fitness of a individual.
    """
    return np.exp(1.0 - (avg_potential_energy(position_list) ** e(i) / avg_potential_energy(position_list_triangular)))

# Gerar um numero aleatorio em binario

def rng_bin(n):
    """
    Takes a int which represents the length of the final binary number.
    Returns a string which represents a number in binary where each char was randomly generated and has lenght n.
    """
    num = ""
    for i in range(n):
        if rng.random() < 0.5:
            num += "0"
        else:
            num += "1"
    return num

def main():

    # Define Probabilities

    pc = 0.1        # Crossover probability
    pm = 0.05       # Mutation probability
    convergence_value = 1000    

    # Generate initial population

    offsprings_decimal = []        #Lista com os individuos todos utilizando valores decimais
    offsprings = []                #Lista igual a de cima mas em binario
    n_ind = 10                    #Numero de individuos, tem de ser par
    rng.seed(1879723891)           #Seed para o gerador de numeros aleatorios, deixar igual por agora para testar programa sem ter imprevistos
    #rng.seed(datetime.now())      #Seed gerada a partir da data em que o programa eh executado, bom para ter alguma randomness no programa mas nao para testar

    for i in range(0,n_ind):
        ind_bin = []
        ind = []
        ind_bin.append(rng_bin(5))
        ind.append((int(ind_bin[0], 2) + 1) / 32)
        ind_bin.append(rng_bin(7))
        ind.append((np.pi / 2.0) * (int(ind_bin[1], 2) + 1) / 128)
        for j in range(0,6):
            ind_bin.append(rng_bin(5))
            ind.append((int(ind_bin[j+2], 2) + 1) / 32)
        offsprings.append(ind_bin)
        offsprings_decimal.append(ind)

    # Compute fitness

    Fitness_list = []

    for ind in offsprings_decimal:
        positions = structure(x(ind),ind)
        positions_triangular = structure(triangular_lat(ind),ind)
        Fitness_list.append(Fitness(positions,positions_triangular,1))

    # Cycle
    
    n_gen_max = 10

    for generation in range(1,n_gen_max+1):

        # Reproduction
        
        total_fitness = sum(Fitness_list)
        relative_fitness = [fit_value/total_fitness for fit_value in Fitness_list]
        offsprings = rng.choices(offsprings, weights=relative_fitness, k=len(offsprings))
        
        # Crossover
        
        offsprings_available = offsprings[:]

        for i in range(len(offsprings)):
            if (rng.random() < pc) and (offsprings[i] in offsprings_available):
                ind_1 = offsprings[i]
                offsprings_available.remove(ind_1)
                ind_2 = rng.choice(offsprings_available)
                offsprings_available.remove(ind_2)
                for j in range(len(ind_1)):
                    gene_1 = ind_1[j]
                    gene_2 = ind_2[j]
                    cross_site = rng.randint(1,len(gene_1) - 1)
                    ind_1[j] = gene_1[:cross_site] + gene_2[cross_site:]
                    ind_2[j] = gene_2[:cross_site] + gene_1[cross_site:]
                offsprings[i] = ind_1
                offsprings[offsprings.index(ind_2)] = ind_2
                
        # Mutation
        
        for i in range(len(offsprings)):
            for j in range(len(offsprings[i])):
                gene = offsprings[i][j]
                for icounter in range(len(gene)):
                    if rng.random() < pm:
                        temp_gene_list = list(gene)
                        if temp_gene_list[icounter] == "0":
                            temp_gene_list[icounter] = "1"
                        else:
                            temp_gene_list[icounter] = "0"
                        gene = "".join(temp_gene_list)
                offsprings[i][j] = gene
                    
        # Compute fitness
        
        offsprings_decimal = []
        for ind in offsprings:
            offsprings_decimal.append([(int(ind[0], 2) + 1) / 32,              # x
                        (np.pi / 2.0) * (int(ind[1], 2) + 1) / 128,      # theta
                        (int(ind[2], 2) + 1) / 32,                       # c21
                        (int(ind[3], 2) + 1) / 32,                       # c22
                        (int(ind[4], 2) + 1) / 32,                       # c31
                        (int(ind[5], 2) + 1) / 32,                       # c32
                        (int(ind[6], 2) + 1) / 32,                       # c41
                        (int(ind[7], 2) + 1) / 32,])                     # c42

        Fitness_list = []

        for ind in offsprings_decimal:
            positions = structure(x(ind),ind)
            positions_triangular = structure(triangular_lat(ind),ind)
            Fitness_list.append(Fitness(positions,positions_triangular,generation))
        
        # Stop when population has converged

        if max(Fitness_list) >= convergence_value:
            break

    best_individual = offsprings_decimal[Fitness_list.index(max(Fitness_list))]

    print(best_individual)

# Executar Programa

if __name__ == '__main__':
    main()


"""
Template for an indivual.

#                      x      theta     c21     c22     c31     c32     c41     c42
ind_bin_template = ["00001","0000001","00001","00001","00001","00001","00001","00001"]

ind_template = [(int(ind_bin[0], 2) + 1) / 32,              # x
       (np.pi / 2.0) * (int(ind_bin[1], 2) + 1) / 128,      # theta
       (int(ind_bin[3], 2) + 1) / 32,                       # c21
       (int(ind_bin[4], 2) + 1) / 32,                       # c22
       (int(ind_bin[5], 2) + 1) / 32,                       # c31
       (int(ind_bin[6], 2) + 1) / 32,                       # c32
       (int(ind_bin[7], 2) + 1) / 32,                       # c41
       (int(ind_bin[8], 2) + 1) / 32,]                      # c42
"""