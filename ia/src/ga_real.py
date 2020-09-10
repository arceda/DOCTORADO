import numpy as np
import random
#random.seed(1)
from numpy.random import seed
import pandas as pd
#seed(1)
np.set_printoptions(precision=4, suppress=True)

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

  print("best candidate: ", population[best_candidate], "\n" )
  return best_candidate

def compute_fitness(population):
    #print(population.shape)
    i = 0
    for chromosome in population: 
        x = chromosome[0]
        y = chromosome[1]

        fitness = (x + 2*y -7)**2 + (2*x + y - 5)**2
        population[i,2] = fitness  
        i += 1
   
###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 20
num_genes = 2 # tenemos numeros de 0 a 511 => necesitamos binarios de 10 bits
n = 15 # number of matting pool
iterations = 3000
torneo_n = 2
prop_crossover = 0.7
BLX_alpha = 0.5
prop_mutation = 0.05
limit_a = -10
limit_b = 10
###########################################################################################################
###########################################################################################################


print("parameters.....................................................")
print("population_size:", population_size)
print("num_genes:", num_genes)
print("matting pool sie:", n)
print("iterations:", iterations)
print("Torneo N:", torneo_n)
print("BLX alpha:", BLX_alpha)
print("prop_crossover:", prop_crossover)
print("prop_mutation Uniform:", prop_mutation)
print("...............................................................\n")

population = np.random.uniform(limit_a, limit_b, size=(population_size, num_genes + 1))# the last column is fitness

print("initial population")
print(population[:,0:2])

print("compute fitness...")
compute_fitness(population)
population_df = pd.DataFrame(data=population, columns=["x", "y", "fitness"])
print(population_df)



for iter in range(iterations):
    print("\n\nITERATION ", iter , "...........................................................................")
    ###########################################################################################################
    ###########################################################################################################
    print(population_df)

    new_population = np.zeros( (n, num_genes + 1) )

    # Matting pool usando Torneo
    print("\ncreating matting pool...........................................") 
    matting_pool = []

    population_size = population.shape[0]

    while len(matting_pool) < n:        
        matting_pool.append( torneo(population, torneo_n) )
    print("matting_pool", matting_pool)
    ###########################################################################################################
    ###########################################################################################################


    for i in range(n):
        ###########################################################################################################
        ###########################################################################################################
        # Cruzamiento
        print("\nCrossover.......................................................") 
        r = random.random()
        father_index = matting_pool[random.randint(0, n - 1)]
        mother_index = matting_pool[random.randint(0, n - 1)]
        father = population[ father_index, 0:num_genes]
        mother = population[ mother_index, 0:num_genes]
        print("father:", father, " mother:", mother)
        if r <= prop_crossover:            
            betha1 = np.random.uniform(-BLX_alpha, 1 + BLX_alpha)     
            betha2 = np.random.uniform(-BLX_alpha, 1 + BLX_alpha)     
            print("yes crossover...", " betha1:", betha1, " betha2:", betha2)   
            #c1 = father[0,0] + betha1*( mother[0,0] - father[0,0] )
            #c2 = father[0,1] + betha2*( mother[0,1] - father[0,1] )
            c1 = father[0] + betha1*( mother[0] - father[0] )
            c2 = father[1] + betha2*( mother[1] - father[1] )
            child = [c1, c2]
        else:        
            if population[father_index, num_genes] > population[mother_index, num_genes]:
              print("not crossover, father have best fitness ", population[father_index, num_genes], population[mother_index, num_genes])
              child = father
            else:
              print("not crossover, mother have best fitness ", population[father_index, num_genes], population[mother_index, num_genes])
              child = mother
        
        child = np.array(child)
        print("child:", child)
        ###########################################################################################################
        ###########################################################################################################



        ###########################################################################################################
        ###########################################################################################################
        # Mutacion
        print("\nMutation........................................................") 
        r = random.random()
        if r <= prop_mutation:        
            pos = random.randint(0, num_genes - 1) 
            new_gene = np.random.uniform(limit_a, limit_b)
            #print(new_gene, child.shape)
            #child[0,pos] = new_gene
            child[pos] = new_gene
            print("yes mutation child: ", child, " pos:", pos) # mutation bit flip
        else:
          print("not mutation")
        ###########################################################################################################
        ###########################################################################################################

        #new_population.append(child[0].tolist())
        new_population[i,0:2] = child


    population = new_population
    compute_fitness(population)
    population_df = pd.DataFrame(data=population, columns=["x", "y", "fitness"])


###########################################################################################################
###########################################################################################################

print("..................................................")
print("\nRESULT..........................................")
print("True result after partial derivates of f(x, y) = f(1, 3) =", 0)
print("Genetic algorithm result:",  np.min(population[:, num_genes]))

import matplotlib.pyplot  as plt

#in x = 1 and y = 3, z is minmum = 0

x = np.outer(np.linspace(-10, 10, 1000), np.ones(1000))
#x = np.outer(np.linspace(-2, 2, 5), np.ones(5))
y = x.copy().T
z = ( x + 2*y - 7 )**2 + (2*x + y - 5)**2

print("result when plotting the surface:",  np.min(z) )


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
plt.show()
