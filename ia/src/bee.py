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
    
    f = (x + 2*y -7)**2 + (2*x + y - 5)**2
    #f = (x)**2 + (y)**2
    if f >= 0:
        fit = 1/( 1 + f )
    else:
        fit = 1 + math.abs( f )
    return f, fit

def compute_fitness(population):
    for sample in population: 
        f, fit = sample_fitness(sample)       
        sample[vector_size] = f
        sample[vector_size + 1] = fit
        

def compute_porbabilities(population):
     # compute propbabilities ##################################################################
    fit_acc = np.sum(population[:, vector_size + 1])
    population[:, vector_size + 3] = population[:, vector_size + 1]/fit_acc
    population[0, vector_size + 4] = population[0, vector_size + 3]
    for i in range( 1, population_size ):
        population[i, vector_size + 4] = population[i, vector_size + 3] + population[i-1, vector_size + 4]

    population_df = pd.DataFrame(data=population, columns=['x', 'y', 'f', 'fit', 'count', 'p', 'p_acum'])
    print("\nbest solutions and probabilities:\n", population_df)

def get_k_j_phi(index):
    k = random.randint(0,population_size-1)
    while k == index:
        k = random.randint(0,population_size-1)

    j = random.randint(0,vector_size-1)
    phi = random.uniform(-1, 1)

    return k, j, phi

###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 10
vector_size = 2 
l = population_size*vector_size
iterations = 1000
limit_a = -10
limit_b = 10
###########################################################################################################
###########################################################################################################

print("parameters.....................................................")
print("SN:", population_size)
print("D:", vector_size)
print("l:", l)
print("Iterations:", iterations)
print("...............................................................\n")

population = np.random.uniform(limit_a, limit_b, size=(population_size, vector_size + 5))
compute_fitness(population)
population[:, 4:7] = np.zeros((population_size,3))

best_solution_index = np.where( population[:, 2] == np.amin(population[:, 2]))
best_solution = population[best_solution_index[0][0], :]

population_df = pd.DataFrame(data=population, columns=['x', 'y', 'f', 'fit', 'count', 'p', 'p_acum'])
print("Population:\n", population_df)

print("\nBesy Solution:", best_solution[0:4])



for iter in range(iterations):
    print("\n***  ITERATION ", iter, "**************************************************************************")
    print("********************************************************************************************\n")

    new_population = population.copy()
    params_ = np.zeros((population_size, 3))
    improve_ = np.ones((population_size, 1))
    
    for i in range( population_size ):
        #print( "\Solution ", i, "->", population[i], "***********************************" )
        
        k, j, phi = get_k_j_phi(i)        
        params_[i] = np.array([k, j, phi]) # solo para guardar como historico

        # equation -> V_i_j = X_i_j + phi ( X_i_j - X_k_j )
        new_population[i, j] = new_population[i, j] + phi * (  new_population[i, j] - new_population[k, j])
        
        old_fitness = new_population[i, vector_size+1]
        f, fit = sample_fitness(new_population[i, 0:2])
        new_population[i, vector_size] = f
        new_population[i, vector_size + 1] = fit

        if fit < old_fitness:
            new_population[i, vector_size + 2] = new_population[i, vector_size + 2] + 1 #update count
            population[i, vector_size + 2] = population[i, vector_size + 2] + 1 #update count
            improve_[i, 0] = 0 # solo para guardar como historico            

        else: # si mejoro
            population[i] = new_population[i]            

    new_population = np.hstack( (params_, new_population) )
    new_population = np.hstack( (new_population, improve_) )
    new_population_df = pd.DataFrame(data=new_population, columns=['k', 'j', 'phi', 'x_', 'y_', 'f', 'fit', 'count', 'p', 'p_acum', 'improve'])
    print("After worker bees:\n", new_population_df)
    
    
    compute_porbabilities(population)


    #############################################################################################
    # observer bees #############################################################################
    #############################################################################################

    for i in range( population_size ):
        bee_i = int(random.choices(range(population_size), population[:, vector_size + 3])[0])
        k, j, phi = get_k_j_phi(bee_i)

        new_solution = population[bee_i, :].copy()
        new_solution[j] = population[bee_i, j] + phi * (  population[bee_i, j] - population[k, j])
        
        f, fit = sample_fitness(new_solution[0:2])
        new_solution[vector_size] = f
        new_solution[vector_size + 1] = fit

        print("\nObserver ", i, " -> i=", bee_i, "k=", k, "j=", j, "; new sol:", new_solution[0:4])
        if fit > population[bee_i, vector_size + 1]: # si mejora
            print("Solution improve")
            population[bee_i] = new_solution
            population[bee_i, vector_size + 2] = 0 # contador = 0
        else:
            print("Solution NOT improve")
            population[bee_i, vector_size + 2] += 1 # incrementamos el contador

                
        compute_porbabilities(population)

    #############################################################################################
    # check empty food                   ########################################################
    #############################################################################################
    for i in range( population_size ):
        if population[i, vector_size + 2] >= l:             
            new_solution = np.random.uniform(limit_a, limit_b, size=(vector_size + 5))
            new_solution[vector_size:vector_size + 5] = np.zeros(5)
            f, fit = sample_fitness(new_solution[0:2])
            new_solution[vector_size] = f
            new_solution[vector_size + 1] = fit
            population[i] = new_solution
            print("solution ", i, "empty food", " new solution = ", new_solution)
    

    population_df = pd.DataFrame(data=population, columns=['x', 'y', 'f', 'fit', 'count', 'p', 'p_acum'])
    print("\nNew population:\n", population_df)
    best_solution_index = np.where( population[:, 2] == np.amin(population[:, 2]))
    best_solution = population[best_solution_index[0][0], :]
    print("\nBesy Solution:", best_solution[0:4])


population = population[population[:, vector_size].argsort()]
population_df = pd.DataFrame(data=population, columns=['x', 'y', 'f', 'fit', 'count', 'p', 'p_acum'])
print("\nLast population:")
print(population_df)

print("..................................................")
print("\nRESULT..........................................")
print("Best fitness:",  population[0, vector_size])
print("Solution:",  population[0, 0:2])
