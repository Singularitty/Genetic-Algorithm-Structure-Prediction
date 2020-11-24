import sys
import importlib
from datetime import datetime
import random as rng
import numpy as np

import settings

convergence_value = settings.convergence_value
n_gen_max = settings.n_gen_max
n_ind = settings.n_ind
pc = settings.pc
pm = settings.pm


import structure_generator as sg


interaction_potential = settings.potential
model = importlib.import_module(interaction_potential)
r_cut = settings.r_cut

# Ground State Calculation

def r(position_1,position_2):
    """
    Takes two tuples representin the positions of two particles.
    Returns a float represeting the distance in space of those two particles.
    """
    return np.sqrt((position_1[0]-position_2[0])**2 + (position_1[1]-position_2[1])**2)

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
                energy_sum += model.potential_energy(distance)
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
    return np.exp(1.0 - (avg_potential_energy(position_list) 
                        / avg_potential_energy(position_list_triangular))**e(i))

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

    # Generate initial population

    offsprings_decimal = []        #Lista com os individuos todos utilizando valores decimais
    offsprings = []                #Lista igual a de cima mas em binario
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
        positions = sg.structure(sg.x(ind),ind)
        positions_triangular = sg.structure(sg.triangular_lat(ind),ind)
        Fitness_list.append(Fitness(positions,positions_triangular,1))

    # Cycle

    for generation in range(1,n_gen_max+1):

        # Reproduction
        
        total_fitness = sum(Fitness_list)
        try:
            relative_fitness = [fit_value/total_fitness for fit_value in Fitness_list]
            offsprings = rng.choices(offsprings, weights=relative_fitness, k=len(offsprings))
        except:
            print(total_fitness)
            print(relative_fitness)
            print(offsprings)
            sys.exit()

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
            positions = sg.structure(sg.x(ind),ind)
            positions_triangular = sg.structure(sg.triangular_lat(ind),ind)
            Fitness_list.append(Fitness(positions,positions_triangular,generation))
        
        # Stop when population has converged

        if max(Fitness_list) >= convergence_value:
            break

    best_individual = offsprings_decimal[Fitness_list.index(max(Fitness_list))]
    print(max(Fitness_list))
    # Print da rede

    with open("rede.xyz","w") as output:
        positions = sg.structure(sg.x(best_individual),best_individual)
        output.write(str(len(positions)) + '\n')
        output.write("\n")
        for particle in positions:
            output.write(str(particle[0]) + " " + str(particle[1]) + " 0\n")

    print(best_individual)

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