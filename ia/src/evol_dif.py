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
    a = sample[2]
    b = sample[3]
    return (x + 2*y -7)**2 + (2*x + y - 5)**2 + (a + 2*b - 7)**2 + (2*a + b - 5)**2

def compute_fitness(population):
    for sample in population:        
        sample[vector_size] = sample_fitness(sample)
        

###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 10
vector_size = 4 
mutation_const = 0.8
crossover_const = 0.5
limit_a = -10 # limits for each attribute in vectors
limit_b = 10
iterations = 200
###########################################################################################################
###########################################################################################################

print("parameters.....................................................")
print("population_size:", population_size)
print("dimensions:", vector_size)
print("Mutation const (F):", mutation_const)
print("Corssover const (CR):", crossover_const)
print("Iterations:", iterations)
print("...............................................................\n")

population = np.random.uniform(limit_a, limit_b, size=(population_size, vector_size + 1))
compute_fitness(population)

population_df = pd.DataFrame(data=population, columns=['x', 'y', 'a', 'b', 'fitness'])
print(population_df)

for iter in range(iterations):
    print("\n***  ITERATION ", iter, "**************************************************************************")
    print("********************************************************************************************\n")

    new_population = np.zeros((population_size, vector_size + 1))

    samples_random = list(range(0, population_size))
    for i in range(population.shape[0]):
        sample = population[i, 0:vector_size]
        old_fitness = population[i, vector_size]

        print( "\nTarget vector:", sample, "*******************************************")        
        random.shuffle(samples_random)
        xm = samples_random[0]
        xl = samples_random[1]
        xk = samples_random[2]       

        # MUTATION ######################################################################
        print("\nMUTATION")
        print("xm=", xm, "xl=", xl, "xk=", xk)
        diff_weighted = mutation_const * ( population[xk] - population[xl] )
        print("F*(xk - xl) (differences weighted) =",  diff_weighted[0:vector_size] ) 
        mutation = diff_weighted + population[xm]
        print("xm + F*(xk - xl) (mutated vector) =", mutation[0:vector_size])

        # CROSSOVER #####################################################################
        print("\nCROSSOVER")
        trial_vector = np.zeros(vector_size)
        aleatory_numbers = np.random.rand(vector_size)
        print("Aleatory numbers for crossover:", aleatory_numbers)
        for j  in range(vector_size):            
            if aleatory_numbers[j] >= crossover_const: # from target vector
                trial_vector[j] = sample[j]
            else: # from mutation vector
                trial_vector[j] = mutation[j]

        print("Trial vector:", trial_vector)
        new_fitness = sample_fitness(trial_vector)

        if new_fitness < old_fitness:
            print("new fitness:", new_fitness, " old fitness:", old_fitness, " => trial vector to new population")
            new_population[i, 0:vector_size] = trial_vector
            new_population[i, vector_size] = new_fitness
        else:
            print("new fitness:", new_fitness, " old fitness:", old_fitness, " => target vector to new population")
            new_population[i, 0:vector_size] = sample
            new_population[i, vector_size] = old_fitness

    
    population = new_population
    population_df = pd.DataFrame(data=population, columns=['x', 'y', 'a', 'b', 'fitness'])
    print("\nNew population:")
    print(population_df)









population = population[population[:, vector_size].argsort()]
population_df = pd.DataFrame(data=population, columns=['x', 'y', 'a', 'b', 'fitness'])
print("\nLast population:")
print(population_df)

print("..................................................")
print("\nRESULT..........................................")
print("Best fitness:",  population[0, vector_size])
print("Solution:",  population[0, 0:vector_size])


