import numpy as np
import random
import scipy.stats
import pandas as pd
import sys
np.set_printoptions(precision=3, suppress=True)
import math

#random.seed(1)
from numpy.random import seed
#seed(1)


def clone(pop):
    pop_cloned = []    
    for i in range(len(priorities)):
        for j in range( priorities[i] ):
            pop_cloned.append( pop[i] )   
    return np.array( pop_cloned )

def mut(sample, num_mutation):
    #print( sample, num_mutation )
    indexes = list(range( sample.shape[0] - 1 ))
    random.shuffle(indexes)

    sample_tmp = sample.copy()
    cache = []

    print("\nsample:", sample[0:vector_size], end=" ")

    i = 0
    num_mutation_count = 0
    print("indexes:", end=' ')
    while num_mutation_count < num_mutation:        
        tmp = sample_tmp[indexes[i]]
        sample_tmp[ indexes[i] ] = sample_tmp[ indexes[i + 1] ]
        sample_tmp[ indexes[i + 1] ] = tmp

        print("[", indexes[i], indexes[i+1] , "]", end= '')
        cache.append( [indexes[i], indexes[i+1]] )

        num_mutation_count += 1
        i += 2

    print(" mutated:", sample_tmp[0:vector_size], end="")

    return sample_tmp, np.array(cache)

def mutation(pop):
    print("\nMutations: ")    
    num_mutation = 1
    index = 0
    for i in range(len(priorities)):        
        for j in range( priorities[i] ):
            sample_mutated, cache = mut( pop[ index ], num_mutation)
            pop[index] = sample_mutated

            index += 1            
        num_mutation += 1

    print()

    

def compute_fitness(population, DIST):   
    for chromosome in population: 
      total_dist = 0
      for i in range(chromosome.shape[0] - 2):
        #print(chromosome[i], " , ", chromosome[i+1], " = ", DIST[chromosome[i], chromosome[i+1]])
        total_dist += DIST[int(chromosome[i]), int(chromosome[i+1])]
      chromosome[-1] = total_dist
  
        

###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 7
population_size_f = 5
size_pclone = 15
size_phyper = 15
size_s = 5
size_r = 2
vector_size = 10 
iterations = 400
limit_a = -10
limit_b = 10
###########################################################################################################
###########################################################################################################

print("parameters.....................................................")
print("Population size P:", population_size)
print("Population size F:", population_size_f)
print("pClone:", size_pclone)
print("pHyper:", size_phyper)
print("size S:", size_s)
print("size R:", size_r)
print("vector size:", vector_size)
print("Iterations:", iterations)
print("...............................................................\n")

DIST = np.array([ [ 0,  1,  3,  23, 11, 5,  83, 21, 28, 45 ],
                  [ 1,  0,  1,  18, 3,  41, 20, 61, 95, 58 ],
                  [ 3,  1,  0,  1,  56, 21, 43, 17, 83, 16 ],
                  [ 23, 18, 1,  0,  1,  46, 44, 45, 50, 11 ],
                  [ 11, 3,  56, 1,  0,  1,  93, 38, 78, 41 ],
                  [ 5,  41, 21, 46, 1,  0,  1,  90, 92, 97 ],
                  [ 83, 20, 43, 44, 93, 1,  0,  1,  74, 29 ],
                  [ 21, 61, 17, 45, 38, 90, 1,  0,  1,  28 ],
                  [ 28, 95, 83, 50, 78, 92, 74, 1,  0,  1  ],
                  [ 45, 58, 16, 11, 41, 97, 29, 28, 1,  0  ]])

cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
cities_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

priorities = [5, 4, 3, 2, 1]

population = np.zeros( (population_size, vector_size + 1) )
tmp_sol = np.array(range(DIST.shape[0]))
for i in range(population.shape[0]):
    random.shuffle(tmp_sol)    
    population[i, 0:vector_size ] = tmp_sol

compute_fitness(population, DIST)
population_df = pd.DataFrame(data=population, columns=["city", "city", "city", "city", "city", "city", "city", "city", "city", "city", "cost"])

print("Initial population:")
print(population_df)

for iter in range(iterations):
    print("\n***  ITERATION ", iter, "**************************************************************************")
    print("********************************************************************************************\n")

    population = population[population[:,vector_size].argsort()]
    population_old = population.copy()
    
    print("population:\n", population_df )

    pop_f = population[ 0:population_size_f, : ]
    pop_f_df = pd.DataFrame(data=pop_f, columns=["city", "city", "city", "city", "city", "city", "city", "city", "city", "city", "cost"])
  
    print("\npopulation F:\n", pop_f_df )

    pop_cloned = clone(population)
    pop_cloned_df = pd.DataFrame(data=pop_cloned, columns=["city", "city", "city", "city", "city", "city", "city", "city", "city", "city", "cost"])
    print("\npopulation cloned:\n", pop_cloned_df )

    mutation(pop_cloned)

    compute_fitness(pop_cloned, DIST)
    print("\npopulation cloned mutated:\n", pop_cloned_df )
    
    # sort new population
    pop_cloned = pop_cloned[pop_cloned[:,vector_size].argsort()]

    # population S and R
    population_s_r = np.zeros( (size_s + size_r, vector_size + 1 ))
    population_s_r[0:size_s, :] = pop_cloned[0:size_s, :]    

    for i in range(size_r):
        random.shuffle(tmp_sol)    
        population_s_r[size_s + i, 0:vector_size ] = tmp_sol
    
    compute_fitness(population_s_r, DIST)

    population_s_r_df = pd.DataFrame(data=population_s_r, columns=["city", "city", "city", "city", "city", "city", "city", "city", "city", "city", "cost"])
    print("\nNew population S and R:")
    print(population_s_r_df)

    # new population
    population_s_r = population_s_r[population_s_r[:,vector_size].argsort()]

    new_population = np.zeros( (population_size, vector_size + 1 ))
    new_population[0:5, :] = population_s_r[0:5, :]    
    new_population[5:7, :] = population_old[0:2, :]    

    population = new_population
    
    population_df = pd.DataFrame(data=population, columns=["city", "city", "city", "city", "city", "city", "city", "city", "city", "city", "cost"])
    print("\nNew population:")
    print(population_df)


population = population[population[:, vector_size].argsort()]
population_df = pd.DataFrame(data=population, columns=["city", "city", "city", "city", "city", "city", "city", "city", "city", "city", "cost"])
print("\nFinal population:")
print(population_df)

print("..................................................")
print("\nRESULT..........................................")
print("Best fitness:",  population[0, vector_size])
print("Solution:",  population[0, 0:vector_size])
