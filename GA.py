import random as rng
from datetime import datetime
import numpy as np

        
# Daoud-Cotton model

global f    # Number of arms in star polymer
f = 50.0

global core_radius  #Core radius
core_radius = 1

global r_cut    #Cutoff radius
r_cut = 2

global a    #Packing fracture
a = 2.4

def daoud_cotton_u(r):
    if r <= core_radius:
        return (5/18)*(f**(1.5)) * (-np.log(r/core_radius) + 1.0 / (1.0 + np.sqrt(f/2)))
    else:
        return ((5/18)*(f**(1.5))) * (core_radius / (1 + np.sqrt(f/2)) * (np.exp((np.sqrt(f)*(r - core_radius))/(2*core_radius)) / r))
    
# Define lattice vectors

def triangular_lat(i,ind):
    if i == 1:
        return (a, 0.0)
    else:
        return (a*ind[0]*0.5, a*ind[0]*(np.sqrt(3)/2))

def x(i,ind):
    if i == 1:
        return (a, 0.0)
    else:
        return (a*ind[0]*np.cos(ind[1]), a*ind[0]*np.sin(ind[1]))


# Positions of b particles in this case 4 total


def y(i,x,ind):
    if i == 1:
        return (0, 0)
    elif i == 2:
        return (ind[i] * x(1)[0] + ind[i + 1] * x(2)[0],
        ind[i + 1] * x(2)[1])
    elif i == 3:
        return (ind[i + 1] * x(1)[0] + ind[i + 2] * x(2)[0],
        ind[i + 2] * x(2)[1])
    else:
        return (ind[i + 2] * x(1)[0] + ind[i + 3] * x(2)[0],
        ind[i + 3] * x(2)[1])

# Ground State Calculation


def r(i, j, x, ind):
    return np.sqrt( (y(i,x, ind)[0]-y(j,x, ind)[0])**2 + (y(i,x, ind)[1]-y(j,x, ind)[1])**2 )


def avg_potential_energy(n, x, ind):
    energy_sum = 0
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            if r(i, j, x, ind) < r_cut:
                energy_sum += daoud_cotton_u(r(i, j, x, ind))
    return (0.5*n)*energy_sum


# Fitness function


def e(i):
    return 1.0 + i * (np.log(i) / 40.0)


def Fitness(n,i,ind):
    return np.exp(1.0 - (avg_potential_energy(n, x, ind) ** e(i) / avg_potential_energy(n, triangular_lat, ind)))


# Individual Template
"""
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

# Gerar um numero aleatorio em binario

def rng_bin(n):
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

    list_ind = []           #Lista com os individuos todos utilizando valores decimais
    list_ind_bin = []       #Lista igual a de cima mas em binario
    n_ind = 100             #Numero de individuos, tem de ser par
    rng.seed(1879723891)    #Seed para o gerador de numeros aleatorios, deixar igual por agora para testar programa sem ter imprevistos
    #rng.seed(datetime.now())   #Seed gerada a partir da data em que o programa eh executado, bom para ter alguma randomness no programa mas nao para testar

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
        list_ind_bin.append(ind_bin)
        list_ind.append(ind)

    # Compute fitness

    Fitness_list = []

    for ind in list_ind:
        Fitness_list.append(Fitness(4,1,ind))

    # Cycle

    offsprings = list_ind_bin[:]
    
    n_gen_max = 1000

    for k in range(1,n_gen_max+1):

        # Reproduction
        
        total_fitness = sum(Fitness_list)
        relative_fitness = [fit_value/total_fitness for fit_value in Fitness_list]
        offsprings = rng.choices(offsprings, weights=relative_fitness, k=len(list_ind_bin))
        
        # Crossover

        for i in range(len(offsprings)):
            if rng.random() < pc:
                ind_1 = offsprings[i]
                ind_2 = rng.choice(offsprings)
                while ind_2 == ind_1:
                    ind_2 = rng.choice(offsprings)
                for j in range(6):
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
                        temp_gene_list = list("gene")
                        if temp_gene_list[icounter] == "0":
                            temp_gene_list[icounter] = "1"
                        else:
                            temp_gene_list[icounter] = "0"
                        gene = "".join(temp_gene_list)
                offsprings[i][j] = gene
                    
        # Compute fitness
        
        offsprings_decimal = [(int(offsprings[0], 2) + 1) / 32,              # x
                     (np.pi / 2.0) * (int(offsprings[1], 2) + 1) / 128,      # theta
                     (int(offsprings[3], 2) + 1) / 32,                       # c21
                     (int(offsprings[4], 2) + 1) / 32,                       # c22
                     (int(offsprings[5], 2) + 1) / 32,                       # c31
                     (int(offsprings[6], 2) + 1) / 32,                       # c32
                     (int(offsprings[7], 2) + 1) / 32,                       # c41
                     (int(offsprings[8], 2) + 1) / 32,]                      # c42

        Fitness_list = []

        for ind in offsprings_decimal:
            Fitness_list.append(Fitness(4,1,ind))
        
        # Stop when population has converged

        if max(Fitness_list) >= convergence_value:
            break

    best_individual = offsprings_decimal[Fitness_list.index(max(Fitness_list))]

    print(best_individual)

# Executar Programa

if __name__ == '__main__':
    main()