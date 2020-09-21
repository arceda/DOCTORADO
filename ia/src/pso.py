import numpy as np
import random
import scipy.stats
import pandas as pd
import sys
np.set_printoptions(precision=4, suppress=True)

#random.seed(1)
from numpy.random import seed
#seed(1)

def sample_fitness(sample):
    x = sample[0]
    y = sample[1]
    #a = sample[2]
    #b = sample[3]
    #return (x + 2*y -7)**2 + (2*x + y - 5)**2 + (a + 2*b - 7)**2 + (2*a + b - 5)**2
    return (x + 2*y -7)**2 + (2*x + y - 5)**2 

def compute_fitness(population):
    for sample in population:        
        sample[vector_size] = sample_fitness(sample)
        

###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 6
vector_size = 4 
phi_1 = 2
phi_2 = 2
iterations = 100
limit_a = -10
limit_b = 10
###########################################################################################################
###########################################################################################################

print("parameters.....................................................")
print("population_size:", population_size)
print("initial values v_i:  [-1,1]")
print("w = [0.1]")
print("rand_1 and rand_2 = [0.1]")
print("phi_1 and phi_2 = 2")
print("Iterations:", iterations)
print("...............................................................\n")

positions = np.random.uniform(limit_a, limit_b, size=(population_size, 2))
velocities = np.random.uniform(-1, 1, size=(population_size, 3))
population = np.hstack( (positions, velocities) )

compute_fitness(population)

population = population[population[:, vector_size].argsort()]

best_locals = population[ :, 0:2 ].copy()
best_locals = np.hstack( ( best_locals, population[ :, vector_size ].reshape( population_size, 1 ).copy() ) )

best_global = best_locals[0]

population_df = pd.DataFrame(data=population, columns=['x', 'y', 'v_x', 'v_y', 'fitness'])
best_locals_df = pd.DataFrame(data=best_locals, columns=['x', 'y', 'fitness'])
print("population:\n", population_df)
print("\nbest locals:\n", best_locals_df)
print("\nbest global:", best_global)

for iter in range(iterations):
    print("\n***  ITERATION ", iter, "**************************************************************************")
    print("********************************************************************************************\n")

    for i in range( population_size ):
        print( "\nParticle ", i, "=", population[i], "***********************************" )
        w = random.random()
        rand_1 = random.random()
        rand_2 = random.random()

        print("w:", w, "rand 1:", rand_1, "rand 2:", rand_2)

        v_current = population[i, 2:4]
        p_best = best_locals[i, 0:2]
        x = population[i, 0:2]
        g = best_global[0:2]
        v_next = w*v_current + phi_1 * rand_1 * ( p_best - x ) + phi_2 * rand_2 * ( g - x )
        x_next = x + v_next
        new_fitness = sample_fitness( x_next )

        print("new position:", x_next, "new velocity:", v_next, "new fitness:", new_fitness)

        if new_fitness < population[i, 4]:
            best_locals[i, 0:2] = x_next
            best_locals[i, 2] = new_fitness

        if new_fitness < best_global[2]:
            best_global[0:2] = x_next
            best_global[2] = new_fitness

        population[i, 0:2] = x_next 
        population[i, 2:4] = v_next
        population[i, 4] = new_fitness

         
    population_df = pd.DataFrame(data=population, columns=['x', 'y', 'v_x', 'v_y', 'fitness'])
    print("\nPopulation:")
    print(population_df)

    best_locals_df = pd.DataFrame(data=best_locals, columns=['x', 'y', 'fitness'])
    print("\nbest locals:\n", best_locals_df)
    print("\nbest global:", best_global)


population = population[population[:, vector_size].argsort()]
population_df = pd.DataFrame(data=population, columns=['x', 'y', 'a', 'b', 'fitness'])
print("\nLast population:")
print(population_df)

print("..................................................")
print("\nRESULT..........................................")
print("Best fitness:",  best_global[2])
print("Solution:",  best_global[0:2])