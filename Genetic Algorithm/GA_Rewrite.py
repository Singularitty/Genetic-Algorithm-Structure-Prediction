import sys
import multiprocessing as mp
import importlib
from datetime import datetime
import random as rng
import numpy as np

import settings

convergence_value = settings.convergence_value
n_gen_max = settings.n_gen_max
n_individuals = settings.n_ind
pc = settings.pc
pm = settings.pm

import Structure as st

interaction_potential = settings.potential
InteractionModel = importlib.import_module(interaction_potential)
r_cut = settings.r_cut

rng.seed(1879723891)
#rng.seed(datetime.now())

# Ground State Calculation

def r(position1, position2):
    """
    Takes two tuples representin the positions of two particles.
    Returns a float represeting the distance in space of those two particles.
    """
    x = position1[0] - position2[0]
    y = position1[1] - position2[1]
    return np.sqrt(x**2 + y**2)

def avg_potential_energy(position_list):
    """
    Takes a list with tuples which represent the positions of particles in space.
    Returns a float representing the average potential energy for a structure made of those particles for a given potential.
    """
    energy_sum = 0
    n = len(position_list)
    for i in range(1, n):
        for j in range(i+1, n):
            distance = r(position_list[i], position_list[j])
            if distance < r_cut:
                energy_sum += InteractionModel.potential_energy(distance)
    return (0.5*n)*energy_sum


# Fitness function

def fitness(structure, triangular_structure, generation):
    """
    Takes a list with tuples representing the positions of particles, takes a int representing the current generation of individuals.
    Returns a float, representing the Fitness of a individual.
    """
    ratio = (avg_potential_energy(structure) / avg_potential_energy(triangular_structure))
    e = (1.0 + generation * (np.log(generation) / 40.0))
    return np.exp(1.0 - ratio**e)

# Gerar um numero aleatorio em binario

def RngBinStr(n):
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

def ComputeFitness(ind, generation):
    structure = st.Structure(st.Lattice(ind),ind)
    triangular_structure = st.Structure(st.TriangularLattice(ind), ind)
    return fitness(structure, triangular_structure, generation)

def BinaryToDecimal(ind):
    return ([(int(ind[0], 2) + 1) / 32,                       # x
             (np.pi / 2.0) * (int(ind[1], 2) + 1) / 128,      # theta
             (int(ind[2], 2) + 1) / 32,                       # c21
             (int(ind[3], 2) + 1) / 32,                       # c22
             (int(ind[4], 2) + 1) / 32,                       # c31
             (int(ind[5], 2) + 1) / 32,                       # c32
             (int(ind[6], 2) + 1) / 32,                       # c41
             (int(ind[7], 2) + 1) / 32,])                     # c42

def main():

    # Generate initial population

    IndividualDecimal = np.dtype([("x",float),
                                  ("theta",float),
                                  ("c21",float),
                                  ("c22",float),
                                  ("c31",float),
                                  ("c32",float),
                                  ("c41",float),
                                  ("c42",float)])



    populationBinary = [[RngBinStr(5) if j != 1 else RngBinStr(7) for j in range(8)] for i in range(n_individuals)]

    populationDecimal = np.empty(n_individuals, dtype=IndividualDecimal)

    for i in range(n_individuals):
        for j in range(8):
            if j != 1:
                populationDecimal[i][j] = (int(populationBinary[i][j], 2) + 1) / 32
            else:
                populationDecimal[i][j] = (np.pi / 2.0) * (int(populationBinary[i][j], 2) + 1) / 128


    # Compute fitness

    pool = mp.Pool(mp.cpu_count())

    fitness_list = pool.map(ComputeFitness(generation = 1), populationDecimal)

    pool.close()


    # Cycle

    for gen in range(1, n_gen_max + 1):

        # Reproduction

        total_fitness = sum(fitness_list)

        relative_fitness = [fit_value/total_fitness for fit_value in fitness_list]

        populationBinary = rng.choices(populationBinary, weights=relative_fitness, k=n_individuals)

        # Crossover

        offsprings_available = populationBinary[:]

        for i in range(n_individuals):
            if (rng.random() < pc) and (populationBinary[i] in offsprings_available):
                ind_1 = populationBinary[i]
                offsprings_available.remove(ind_1)
                ind_2 = rng.choice(offsprings_available)
                temp_index = populationBinary.index(ind_2)
                offsprings_available.remove(ind_2)
                for j in range(8):
                    gene_1 = ind_1[j]
                    gene_2 = ind_2[j]
                    cross_site = rng.randint(1, len(gene_1) - 1)
                    ind_1[j] = gene_1[:cross_site] + gene_2[cross_site:]
                    ind_2[j] = gene_2[:cross_site] + gene_1[cross_site:]
                populationBinary[i] = ind_1
                populationBinary[temp_index] = ind_2

        # Mutation

        for i in range(n_individuals):
            for j in range(8):
                gene = populationBinary[i][j]
                for icounter in range(len(gene)):
                    if rng.random() < pm:
                        temp_gene_list = list(gene)
                        if temp_gene_list[icounter] == "0":
                            temp_gene_list[icounter] = "1"
                        else:
                            temp_gene_list[icounter] = "0"
                        gene = "".join(temp_gene_list)
                populationBinary[i][j] = gene

        # Compute fitness

        pool = mp.Pool(mp.cpu_count())

        populationDecimal = pool.map(BinaryToDecimal, populationBinary)

        fitness_list = pool.map(ComputeFitness(generation = gen), populationDecimal)

        pool.close()

        # Stop when population has converged

        if max(fitness_list) >= convergence_value: break

    best_individual = populationDecimal[fitness_list.index(max(fitness_list))]
    print(max(fitness_list))
    print(best_individual)

    st.SaveStructure(best_individual)





# Executar Programa

#import warnings
#import time
if __name__ == '__main__':
    #np.seterr(all="warn")
    #warnings.filterwarnings("error")
    #start_time = time.time()
    main()
    #print("--- %s seconds ---" % (time.time() - start_time))


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
