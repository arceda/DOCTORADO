import numpy as np
import random
import scipy.stats
import pandas as pd

#random.seed(1)
from numpy.random import seed
#seed(1)

def compute_fitness(population):
    fitness_array = []

    i = 0
    for chromosome in population: 
        #print(chromosome, chromosome.shape)
        x = chromosome[0]
        y = chromosome[1]

        fitness = (x + 2*y -7)**2 + (2*x + y - 5)**2
        #fitness_array.append(fitness)

        population[i,2] = fitness

        i += 1
    #print("fitness:", fitness_array)
    #population = np.hstack((population, np.matrix(fitness_array).T ))
    #return population

#def crossover(father, mother):
    
        

###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 1
num_genes = 2 # tenemos numeros de 0 a 511 => necesitamos binarios de 10 bits
n = 1 # number of matting pool
sigma_val = 0.2
iterations = 300
#prop_crossover = 0.7
#BLX_alpha = 0.5
#prop_mutation = 0.05
limit_a = -10
limit_b = 10
###########################################################################################################
###########################################################################################################


print("parameters.....................................................")
print("population_size:", population_size)
print("num_genes:", num_genes)
print("matting pool sie:", n)
print("iterations:", iterations)
#print("BLX alpha:", BLX_alpha)
#print("prop_crossover:", prop_crossover)
#print("prop_mutation Uniform:", prop_mutation)
print("...............................................................\n")

population = np.random.uniform(limit_a, limit_b, size=(population_size, num_genes))
population = np.hstack((population, [[0, sigma_val, sigma_val]]))
population_df = pd.DataFrame(data=population, columns=["x", "y", "fitness", "sigma_x", "sigma_y"])

print("initial population")
print(population_df)

print("\ncompute fitness...")
compute_fitness(population)
print(population_df)


mu = 0 # default for gauss

for i in range(iterations):
    print("\n\nITERATION " + str(i) + "...")

    #print("Population : ")
    print(population_df)

    auc_x = random.random() #area under the curve
    auc_y = random.random() #area under the curve

    sigma_1 = population[0,3]
    sigma_2 = population[0,4]
    
    gaussian_number_1 = scipy.stats.norm(mu, sigma_1).ppf(auc_x) #return x given area under the curve
    gaussian_number_2 = scipy.stats.norm(mu, sigma_2).ppf(auc_y) #return x given area under the curve

    print("Random numbers:", [auc_x, auc_y], " Gauss numbers: ", [gaussian_number_1,gaussian_number_2])

    old_fitness = population[0,2]
    old_x = population[0,0]
    old_y = population[0,1]
    old_sigma_1 = population[0,3]
    old_sigma_2 = population[0,4]
    #print(population)

    #aplicamos la mutacion
    population[0,0] = population[0,0] + gaussian_number_1
    population[0,1] = population[0,1] + gaussian_number_2
    population[:, 3:5] = 1.5*population[:, 3:5]

    compute_fitness(population)
    print("\nmutation and fitness")
    print(population_df)
    #print(old_fitness)

    if population[0,2] < old_fitness:
      print("offpring better")
    else:
      print("offspring worst, we return population to father")
      population[0,0] = old_x
      population[0,1] = old_y
      population[0,2] = old_fitness
      population[0,3] = (1.5**-0.25)*old_sigma_1
      population[0,4] = (1.5**-0.25)*old_sigma_2
      #population[:, 3:5] = (1.5**-0.25)*population[:, 3:5]

       

###########################################################################################################
###########################################################################################################

print("..................................................")
print("\nRESULT..........................................")
print("True result after partial derivates of f(x, y) = f(1, 3) =", 0)
print("Genetic algorithm result:",  np.min(population[:, num_genes]))
