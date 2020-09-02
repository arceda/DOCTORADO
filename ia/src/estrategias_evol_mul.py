import numpy as np
import random
import scipy.stats
import pandas as pd
import math
import sys

#random.seed(1)
from numpy.random import seed
np.set_printoptions(precision=4, suppress=True)
#seed(1)

def get_best_pop(population, new_population):
  total_pop = np.hstack((population, new_population))


def compute_fitness(population):
    fitness_array = []
    i = 0
    for chromosome in population: 
        x = chromosome[0]
        y = chromosome[1]

        fitness = (x + 2*y -7)**2 + (2*x + y - 5)**2
        population[i,2] = fitness
        i += 1

def torneo(population, n):
  
  population_size = population.shape[0]

  # we get the candidates to be fathers
  random_indexes = np.random.randint(population_size, size=(n))  

  #get the ramdom candidates (whole row)
  candidates = np.take(population, random_indexes, axis=0) 

  print("candidates: ")
  print(candidates )

  candidates_fitness = candidates[:, 2] #extract the fitness into a vector
  best_candidate_tmp = candidates_fitness.argmin(axis=0) # obtnemos el indice the candidato con menor fitness
  best_candidate = random_indexes[best_candidate_tmp] #indice dle candidato con menor indice, pero en la population

  print("best candidate: ", population[best_candidate] )
  return best_candidate

  # the same using for
  '''
  for i in range(n):    
    candidate = random.randint(0, population_size - 1) # 0 <= rand <= population_size - 1   
    print("fitness candidate ", candidate , " " , population[candidate,0:2], " is ", population[candidate, 2])
    if i == 0:
      index_best_candidate = candidate
      best_fitness = population[candidate, 2]
    else:       
      if population[candidate, 2] < best_fitness:
        index_best_candidate = candidate
        best_fitness = population[candidate, 2]


  return index_best_candidate

  '''

###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 8
num_genes = 2 # tenemos numeros de 0 a 511 => necesitamos binarios de 10 bits
lambd = 6 # number of matting pool
sigma_val = 0.2
iterations = 50
torneo_n = 2

population_size = int(sys.argv[1])
lambd = int(sys.argv[2])
iterations = int(sys.argv[3])

#prop_crossover = 0.7
#BLX_alpha = 0.5
#prop_mutation = 0.05
limit_a = -10
limit_b = 10
###########################################################################################################
###########################################################################################################


print("parameters.....................................................")
print("population_size (mu):", population_size)
print("num_genes:", num_genes)
print("num offprings(lambda):", lambd)
print("Torneo N:", torneo_n)
print("iterations:", iterations)
#print("BLX alpha:", BLX_alpha)
#print("prop_crossover:", prop_crossover)
#print("prop_mutation Uniform:", prop_mutation)
print("...............................................................\n")

population = np.random.uniform(limit_a, limit_b, size=(population_size, num_genes + 1))
population = np.hstack((population, np.zeros((population_size, num_genes)) + sigma_val ))

population_df = pd.DataFrame(data=population, columns=["x", "y", "fitness", "sigma_x", "sigma_y"])
print("initial population")
print(population_df)

print("\ncompute fitness...")
compute_fitness(population)
print(population_df)


gauss_mean = 0 # default for gauss
delta_sigma = 1.0/( (2*(num_genes**0.5))**0.5 ) 

for i in range(iterations):
    print("\n\n\nITERATION " + str(i) + "...............................................................\n")

    #print("Population : ")
    print(population_df)
    #print(population)
  
  
    new_population = np.zeros( (lambd, 5) )

    population_size = population.shape[0]

    lambda_count = 0
    while lambda_count < lambd:
      print("\nOFFSPRING ..................................... ", lambda_count + 1)

      #########################################################################################
      #########################################################################################
      #Crossover
      print("Torneo N:", torneo_n, " => for father 1")
      father1 = torneo(population, torneo_n)
      print("Torneo N:", torneo_n, "=> for father 2")
      father2 = torneo(population, torneo_n)
      
      print("\ncrossover between: ")
      print("father 1:", population[father1, 0:2], population[father1, 3:5] , " father 2:", population[father2, 0:2], population[father2, 3:5])
      #print(population_df.loc[father1,:])
      #print(population_df.loc[father2,:])

      new_population[ lambda_count, 0] = (population[father1, 0] + population[father2, 0])/2
      new_population[ lambda_count, 1] = (population[father1, 1] + population[father2, 1])/2
      new_population[ lambda_count, 3] = (population[father1, 3]*population[father2, 3])**0.5
      new_population[ lambda_count, 4] = (population[father1, 4]*population[father2, 4])**0.5
      compute_fitness(new_population)

      print(new_population[lambda_count, 0:2], new_population[lambda_count, 3:5])
      #########################################################################################
      #########################################################################################


      #########################################################################################
      #########################################################################################
      #Mutation
      print ("\nMutation...")
      #mutamos sigma 
      auc_x = random.random() #area under the curve
      auc_y = random.random() #area under the curve
      gaussian_number_1 = scipy.stats.norm(gauss_mean, delta_sigma).ppf(auc_x) #return x given area under the curve
      gaussian_number_2 = scipy.stats.norm(gauss_mean, delta_sigma).ppf(auc_y) #return x given area under the curve

      print("Mutamos sigma... Random numbers:", [auc_x, auc_y], " Gauss numbers: ", [gaussian_number_1, gaussian_number_2])
      #sigmas are in new_population[lambda_count,3] and new_population[lambda_count,4]
      new_population[lambda_count,3] = new_population[lambda_count,3] * math.exp(gaussian_number_1)
      new_population[lambda_count,4] = new_population[lambda_count,4] * math.exp(gaussian_number_2)
      
      #mutamos 'x' y 'y'
      sigma_mutado_1 =  new_population[lambda_count,3]
      sigma_mutado_2 =  new_population[lambda_count,4]
      auc_x = random.random() #area under the curve
      auc_y = random.random() #area under the curve
      gaussian_number_1 = scipy.stats.norm(gauss_mean, sigma_mutado_1).ppf(auc_x) #return x given area under the curve
      gaussian_number_2 = scipy.stats.norm(gauss_mean, sigma_mutado_2).ppf(auc_y) #return x given area under the curve

      print("Mutamos gene.... Random numbers:", [auc_x, auc_y], " Gauss numbers: ", [gaussian_number_1, gaussian_number_2])
      #genes are in new_population[lambda_count,0] and new_population[lambda_count,1]
      new_population[lambda_count,0] = new_population[lambda_count,0] + gaussian_number_1
      new_population[lambda_count,1] = new_population[lambda_count,1] + gaussian_number_2
      
      compute_fitness(new_population)

      print(new_population[lambda_count])

      #########################################################################################
      #########################################################################################
      lambda_count += 1  

    # fathers_and_children
    

    # obtenemos los mejores de los padres e hijos
    total_pop = np.vstack((population, new_population))
    print("\nJoin fathers and childrem:")
    print(total_pop)
    total_pop_sorted = total_pop[np.argsort(total_pop[:, 2])]
    population = total_pop_sorted[0:population_size, :]

    population_df = pd.DataFrame(data=population, columns=["x", "y", "fitness", "sigma_x", "sigma_y"])

###########################################################################################################
###########################################################################################################

print("..................................................")
print("\nRESULT..........................................")

print("True result after partial derivates of f(x, y) = f(1, 3) =", 0)
print("Genetic algorithm result:",  np.min(population[:, num_genes]))

