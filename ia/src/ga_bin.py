# genetic algorithm binary example

import numpy as np
import random
#random.seed(1)
from numpy.random import seed
#seed(1)

def compute_fitness(population):
    fitness_array = []
    for chromosome in population: 
        #print(chromosome, chromosome.shape)

        chromosome_str = ""        
        for gene in chromosome:        
            chromosome_str = chromosome_str + str(gene)
        
        #print("chromosome_str", chromosome_str)
        x = int(chromosome_str, 2)    
        fitness = x**2 - 250*x - 25
        fitness_array.append(fitness)

    print("fitness:", fitness_array)
    population = np.hstack((population, np.matrix(fitness_array).T ))
    return population

#def crossover(father, mother):
    
        

###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 10
num_genes = 10 # tenemos numeros de 0 a 511 => necesitamos binarios de 10 bits
n = 10 # number of matting pool
iterations = 50
prop_crossover = 0.7
prop_mutation = 0.95
crossover_point = 3
###########################################################################################################
###########################################################################################################


print("parameters.....................................................")
print("population_size:", population_size)
print("num_genes:", num_genes)
print("matting pool sie:", n)
print("iterations:", iterations)
print("prop_crossover:", prop_crossover)
print("prop_mutation:", prop_mutation)
print("crossover_point:", crossover_point)
print("...............................................................\n")

population = np.random.randint(2, size=(population_size, num_genes))

print("initial population")
print(population, population.shape)

print("compute fitness...")
population = compute_fitness(population)
#print(population, population.shape)


for i in range(iterations):
    
    ###########################################################################################################
    ###########################################################################################################
    # Matting pool usando Torneo
    print("\ncreating matting pool...........................................") 
    matting_pool = []
    while len(matting_pool) < n:
        candidate_1 = random.randint(0, population_size - 1) # 0 <= rand <= population_size - 1 
        candidate_2 = random.randint(0, population_size - 1) # 0 <= rand <= population_size - 1 

        fitness_candidate_1 = population[candidate_1, num_genes]
        fitness_candidate_2 = population[candidate_2, num_genes]
        #print(fitness_candidate_1, fitness_candidate_1)

        if fitness_candidate_1 < fitness_candidate_2: 
            matting_pool.append(candidate_1)
        else:
            matting_pool.append(candidate_2)
    print("matting_pool", matting_pool)
    ###########################################################################################################
    ###########################################################################################################


    new_population= []
    while len(new_population) < n:

        ###########################################################################################################
        ###########################################################################################################
        # Cruzamiento
        print("\nCrossover.......................................................") 
        r = random.random()
        father_index = matting_pool[random.randint(0, n - 1)]
        mother_index = matting_pool[random.randint(0, n - 1)]
        father = population[ father_index, 0:population.shape[1]-1 ]
        mother = population[ mother_index, 0:population.shape[1]-1 ]
        print("father:", father, " mother:", mother, " crossover_point:", crossover_point)
        if r <= prop_crossover:
            print("yes crossover...")        
            child_1 = np.hstack((father[0, 0:crossover_point+1],  mother[0, crossover_point+1:population.shape[1]-1]))
            child_2 = np.hstack((mother[0, 0:crossover_point+1],  father[0, crossover_point+1:population.shape[1]-1]))        
        else:        
            child_1 = father
            child_2 = mother
        print("child_1:", child_1, " child_2:", child_2)
        ###########################################################################################################
        ###########################################################################################################



        ###########################################################################################################
        ###########################################################################################################
        # Mutacion
        print("\nMutation........................................................") 
        r = random.random()
        if r <= prop_mutation:        
            pos = random.randint(0, num_genes - 1) 
            child_1[0,pos] = not child_1[0,pos]
            print("yes mutation child 1: ", child_1, " pos:", pos) # mutation bit flip

        r = random.random()
        if r <= prop_mutation:        
            pos = random.randint(0, num_genes - 1) 
            child_2[0,pos] = not child_1[0,pos]
            print("yes mutation child 2: ", child_2, " pos:", pos) # mutation bit flip
        ###########################################################################################################
        ###########################################################################################################

        new_population.append(child_1[0].tolist()[0])
        new_population.append(child_2[0].tolist()[0])     



    new_population = np.array(new_population)
    print("\nnew_population:")
    print(new_population, new_population.shape)
    population = compute_fitness(new_population)



print("..................................................")
print("\nRESULT..........................................")
print("True result:", -15650)
print("Genetic algorithm result:",  np.min(population[:, num_genes]))
