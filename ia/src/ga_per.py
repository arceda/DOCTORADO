import numpy as np
import random
#random.seed(1)
from numpy.random import seed
import pandas as pd
import sys
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

  candidates_fitness = candidates[:, num_genes] #extract the fitness into a vector
  best_candidate_tmp = candidates_fitness.argmin(axis=0) # obtnemos el indice the candidato con menor fitness
  best_candidate = random_indexes[best_candidate_tmp] #indice dle candidato con menor indice, pero en la population

  print("best candidate: ", population[best_candidate], "\n" )
  return best_candidate

def crossover_PBX(ch1, ch2, n):
  child = np.zeros( ch1.shape[0] )
  
  #get ramdon unique n  positions
  positions = np.array(range(0, ch1.shape[0])) 
  np.random.shuffle(positions)
  positions = positions[0:n]

  print("positions: ", positions)

  #for i in range(positions.shape[0]):
    

def compute_fitness(population, DIST):
    #print(population.shape)
    i = 0
    
    for chromosome in population: 
      total_dist = 0
      for i in range(chromosome.shape[0] - 1):
        #print(chromosome[i], chromosome[i+1])
        total_dist += DIST[chromosome[i], chromosome[i+1]]
      chromosome[-1] = total_dist
      i += 1
   
###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 20
num_genes = 10 # tenemos numeros de 0 a 511 => necesitamos binarios de 10 bits
n = 10 # number of matting pool
iterations = 3000
torneo_n = 2
PBX_n = 4
prop_mutation = 0.05
prop_crossover = 0.9
limit_a = -10
limit_b = 10
###########################################################################################################
###########################################################################################################

DIST = np.array([  [ 0,  1,  3,  23, 11, 5,  83, 21, 28, 45 ],
          [ 1,  0,  1,  18, 3,  41, 20, 61, 95, 58 ],
          [ 3,  1,  0,  1,  56, 21, 43, 17, 83, 16 ],
          [ 23, 18, 1,  0,  1,  46, 44, 45, 50, 11 ],
          [ 11, 3,  56, 1,  0,  1,  93, 38, 78, 41 ],
          [ 5,  41, 21, 46, 1,  0,  1,  90, 92, 97 ],
          [ 83, 20, 43, 44, 93, 1,  0,  1,  74, 29 ],
          [ 21, 61, 17, 45, 38, 90, 1,  0,  1,  28 ],
          [ 28, 95, 83, 50, 78, 92, 74, 1,  0,  1  ],
          [ 45, 58, 16, 11, 41, 97, 29, 28, 1,  0  ]])

print("parameters.....................................................")
print("population_size:", population_size)
print("num_genes:", num_genes)
print("matting pool sie:", n)
print("iterations:", iterations)
print("Torneo N:", torneo_n)
print("Crossover PBX n:", PBX_n)
print("prop_mutation Uniform:", prop_mutation)
print("crossover Uniform:", prop_crossover)
print("...............................................................\n")

solution = np.array(range(0, DIST.shape[0]))

population = np.zeros( (population_size, DIST.shape[0] + 1 ))

for i in range(population.shape[0]):
  np.random.shuffle(solution)
  population[i, 0:DIST.shape[0]] = solution

population = population.astype(int)

print("initial population")
print(population[:,0:DIST.shape[0]])

print("compute fitness...")
compute_fitness(population, DIST)

population_df = pd.DataFrame(data=population, columns=["city", "city", "city", "city", "city", "city", "city", "city", "city", "city", "fitness"])
print(population_df)


for i in range(iterations):
    print("\n\nITERATION ", i , "...........................................................................")
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
            crossover_PBX(father, mother, PBX_n)
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
