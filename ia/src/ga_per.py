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
  child_1 = np.zeros(ch1.shape[0], dtype='int') - 1
  child_2 = np.zeros(ch1.shape[0], dtype='int') - 1

  positions = np.array(range(0, ch1.shape[0])) 
  np.random.shuffle(positions)
  positions = positions[0:n]
  #positions = np.array([0, 1, 2])

  print("positions: ", positions)

  # compute child_1
  ch1_tmp = ch1.copy()
  ch2_tmp = ch2.copy()
  for i in range(positions.shape[0]):
    child_1[positions[i]] = ch2[positions[i]]
    ch1_tmp = np.delete(ch1_tmp, np.where(ch1_tmp == ch2[positions[i]]))

  index = 0
  for i in range(child_1.shape[0]):
    if child_1[i] == -1:
      child_1[i] = ch1_tmp[index]
      index += 1

  # compute child_2
  ch1_tmp = ch1.copy()
  ch2_tmp = ch2.copy()
  for i in range(positions.shape[0]):
    child_2[positions[i]] = ch1[positions[i]]
    ch2_tmp = np.delete(ch2_tmp, np.where(ch2_tmp == ch1[positions[i]]))

  index = 0
  for i in range(child_2.shape[0]):
    if child_2[i] == -1:
      child_2[i] = ch2_tmp[index]
      index += 1

  return child_1, child_2

def compute_fitness(population, DIST):   
    for chromosome in population: 
      total_dist = 0
      for i in range(chromosome.shape[0] - 2):
        #print(chromosome[i], " , ", chromosome[i+1], " = ", DIST[chromosome[i], chromosome[i+1]])
        total_dist += DIST[chromosome[i], chromosome[i+1]]
      chromosome[-1] = total_dist
  
   
###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 20
num_genes = 10 # tenemos numeros de 0 a 511 => necesitamos binarios de 10 bits
n = 20 # number of matting pool
iterations = 1000
torneo_n = 5
PBX_n = 4
prop_mutation = 0.5
prop_crossover = 0.9

###########################################################################################################
###########################################################################################################

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

print("parameters.....................................................")
print("population_size:", population_size)
print("num_genes:", num_genes)
print("matting pool size:", n)
print("iterations:", iterations)
print("Torneo N:", torneo_n)
print("Crossover PBX n:", PBX_n)
print("prop_mutation Uniform:", prop_mutation)
print("crossover Uniform:", prop_crossover)
print("...............................................................\n")

DIST_df = pd.DataFrame(data=DIST, columns=["city 0", "city 1", "city 2", "city 3", "city 4", "city 5", "city 6", "city 7", "city 8", "city 9"], 
index=["city 0", "city 1", "city 2", "city 3", "city 4", "city 5", "city 6", "city 7", "city 8", "city 9"])
print("Distances:\n", DIST_df, "\n\n")



solution = np.array(range(0, DIST.shape[0]))

population = np.zeros( (population_size, DIST.shape[0] + 1 ))

for i in range(population.shape[0]):
  np.random.shuffle(solution)
  population[i, 0:DIST.shape[0]] = solution

population = population.astype(int)

#population[0] = [8, 9, 4, 6, 7, 2, 1, 3, 5, 0, -1]
#population[1] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1]

print("initial population")
print(population[:,0:DIST.shape[0]])



print("compute fitness...")
compute_fitness(population, DIST)

population_df = pd.DataFrame(data=population, columns=["city", "city", "city", "city", "city", "city", "city", "city", "city", "city", "fitness"])
print(population_df)

fitness_history = []

for iter in range(iterations):
    print("\n\nITERATION ", iter , "...........................................................................")
    ###########################################################################################################
    ###########################################################################################################
    print(population_df)

    fitness_history.append (  [ np.mean(population[:, num_genes]), np.min(population[:, num_genes]) ] )

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


    for i in range(0, n, 2):
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
            print("yes crossover")
            child_1, child_2 = crossover_PBX(father, mother, PBX_n)
        else:        
            print("no crossover")
            child_1 = father
            child_2 = mother

        print(child_1, child_2)
           
        ###########################################################################################################
        ###########################################################################################################


        ###########################################################################################################
        ###########################################################################################################
        # Mutacion
        print("\nMutation........................................................") 
        r = random.random()
        if r <= prop_mutation:        
            print("yes mutation")
            pos = [random.randint(0, num_genes - 1), random.randint(0, num_genes - 1) ] 
            tmp = child_1[pos[0]]
            child_1[pos[0]] = child_1[pos[1]]
            child_1[pos[1]] = tmp
            print("pos:", pos, "child_1: ", child_1)

            pos = [random.randint(0, num_genes - 1), random.randint(0, num_genes - 1) ] 
            tmp = child_2[pos[0]]
            child_2[pos[0]] = child_2[pos[1]]
            child_2[pos[1]] = tmp            
            print("pos:", pos, "child_2: ", child_2)
        else:
          print("not mutation")
        ###########################################################################################################
        ###########################################################################################################

        #new_population.append(child[0].tolist())
        new_population[i,0:num_genes] = child_1
        new_population[i+1,0:num_genes] = child_2


        

    population = new_population.astype(int)
    compute_fitness(population, DIST)
    population_df = pd.DataFrame(data=population, columns=["city", "city", "city", "city", "city", "city", "city", "city", "city", "city", "fitness"])


###########################################################################################################
###########################################################################################################

print("\n..................................................")
print("\nRESULT..........................................")
print("Population:")
print(population_df)
print("Genetic algorithm result:",  np.min(population[:, num_genes]))


import matplotlib.pyplot  as plt

fitness_history = np.array(fitness_history)

fig = plt.figure()
ax = plt.axes()

ax.plot(range(fitness_history.shape[0]), fitness_history[:, 0], label="pop_fitness mean")
ax.plot(range(fitness_history.shape[0]), fitness_history[:, 1], label="pop_fitness min")
ax.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')

plt.savefig('fitness_history.png', dpi = 300)