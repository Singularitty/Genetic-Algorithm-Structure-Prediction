#Settings to be used by several functions gathered here
#You can easely change the simulation parameters here

#######################################
#### Choosen Interaction Potential ####
#######################################

available_potentials = ["daoud_cotton_model","elastic_multipole"]

potential = available_potentials[0]

####################################
#### Genetic algorithm settings ####
####################################

#Fitness convergence value
convergence_value = 2.5

#Number of maximum Generations
n_gen_max = 10

#Number of random individuals generated 
n_ind = 10 

#Crossover probability
pc = 0.1

#Mutation probability
pm = 0.05

############################
#### Structure Settings ####
############################


#Packing fracture
a = 0.60


######################################
#### Daoud-Cotton model settings #####
######################################

# Number of arms in star polymer
f = 50.0

#Core radius
core_radius = 1.

#Cutoff radius
r_cut = 2.