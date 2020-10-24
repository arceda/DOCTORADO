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
population_size = 10
vector_size = 4 
alpha = 0.99
gamma = 2.411
loudness = 0.5026
pulse = 0.4205
f_min = 0.0
f_max = 0.5
iterations = 500
limit_a = -10
limit_b = 10
###########################################################################################################
###########################################################################################################

print("parameters.....................................................")
print("Num bats:", population_size)
print("Alpha:", alpha)
print("Gamma:", gamma)
print("Loudness (A):", loudness)
print("Pulse (r):", pulse)
print("f_min:", f_min)
print("f_max:", f_max)
print("Beta (B): Between 0 and 1")
print("Epsilon (e): Between -1 and 1")
print("Iterations:", iterations)
print("...............................................................\n")

positions = np.random.uniform(limit_a, limit_b, size=(population_size, 2))
velocities = np.zeros((population_size, 3))
population = np.hstack( (positions, velocities) )

compute_fitness(population)

population = population[population[:, vector_size].argsort()]

best_global = population[0]

population_df = pd.DataFrame(data=population, columns=['x', 'y', 'v_x', 'v_y', 'fitness'])
print("population:\n", population_df)
print("\nbest global:", best_global)

current_pulse = pulse

for iter in range(iterations):
    print("\n***  ITERATION ", iter, "**************************************************************************")
    print("********************************************************************************************\n")

    for i in range( population_size ):
        print( "\nBat ", i, "->", population[i], "***********************************" )
        new_solution = np.zeros( (vector_size + 1) )
        old_fitness = population[i, vector_size]

        betha = random.random()
        f = f_min + (f_max - f_min) * betha
        # equation (9) -> v_ = v + (x - x_gl)*f ##################################################
        new_solution[2:4 ] = population[i, 2:4 ] + ( population[i, 0:2 ]  - best_global[0:2 ])*f 

        # equation (10) -> x_ = x + v ############################################################
        new_solution[0:2 ] = population[i, 0:2 ] + new_solution[2:4 ]  
        new_solution[vector_size] = sample_fitness(new_solution[0:2])
        
        print("Betha:", betha)
        print("Frecuency:", f)
        print("New velocity:", new_solution[2:4 ] )
        print("New position:", new_solution[0:2 ] )
        print("New fitness:", new_solution[ vector_size ] )        

        if random.random() > current_pulse:
            print("Get into pulse rate")
            epsilon = np.random.uniform(-1, 1, size=(2))

            print("Epsilon:", epsilon)
            # equation (11) -> x_new = x_old + e*A ################################################
            new_solution[0:2] = new_solution[0:2] + epsilon*loudness
            new_solution[vector_size] = sample_fitness(new_solution[0:2])
            print("New solution:", new_solution[0:2], " fitness:", new_solution[vector_size] )

        if random.random() < loudness and new_solution[vector_size] < old_fitness: 
            #print(new_solution[vector_size],  old_fitness)
            print("Update bat's position:", population[i, 0:2], "to", new_solution[0:2])
            population[i] = new_solution

            # equation (12) -> A_ = A alpha*A #####################################################
            loudness = alpha * loudness

            # equation (13) -> r_ = r[1-exp(-gamma*t)] #####################################################
            current_pulse = pulse * (  1 - math.exp(-gamma * iter) )
        else:
            print("NO Update bat's position")

        # check if bat is the best global
        if population[i, vector_size] < best_global[vector_size]:
            print("Update best global")
            best_global = population[i]
        else:
            print("NO update best global")

    
    population_df = pd.DataFrame(data=population, columns=['x', 'y', 'v_x', 'v_y', 'fitness'])
    print("\nNew population:")
    print(population_df)

    


population = population[population[:, vector_size].argsort()]
population_df = pd.DataFrame(data=population, columns=['x', 'y', 'a', 'b', 'fitness'])
print("\nLast population:")
print(population_df)

print("..................................................")
print("\nRESULT..........................................")
print("Best fitness:",  best_global[vector_size])
print("Solution:",  best_global[0:2])