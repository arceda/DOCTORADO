
# genetic algorithm binary example

import numpy as np
import random
#random.seed(1)
from numpy.random import seed
#seed(1)

def compute_fitness(population):
    fitness_array = []


    for chromosome in population: 
        #print(chromosome, chromosome.shape)
        x = chromosome[0]
        y = chromosome[1]

        fitness = (x + 2*y -7)**2 + (2*x + y - 5)**2
        fitness_array.append(fitness)

    print("fitness:", fitness_array)
    population = np.hstack((population, np.matrix(fitness_array).T ))
    return population

#def crossover(father, mother):
    
        

###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 20
num_genes = 2 # tenemos numeros de 0 a 511 => necesitamos binarios de 10 bits
n = 16 # number of matting pool
iterations = 5000
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
print("BLX alpha:", BLX_alpha)
print("prop_crossover:", prop_crossover)
print("prop_mutation Uniform:", prop_mutation)
print("...............................................................\n")

population = np.random.uniform(limit_a, limit_b, size=(population_size, num_genes))

print("initial population")
print(population, population.shape)

print("compute fitness...")
population = compute_fitness(population)
print(population, population.shape)



for i in range(iterations):
    
    ###########################################################################################################
    ###########################################################################################################
    # Matting pool usando Torneo
    print("\ncreating matting pool...........................................") 
    matting_pool = []

    population_size = population.shape[0]

    while len(matting_pool) < n:
        candidate_1 = random.randint(0, population_size - 1) # 0 <= rand <= population_size - 1 
        candidate_2 = random.randint(0, population_size - 1) # 0 <= rand <= population_size - 1 

        fitness_candidate_1 = population[candidate_1, num_genes]
        fitness_candidate_2 = population[candidate_2, num_genes]
        #print(fitness_candidate_1, fitness_candidate_1)

        if fitness_candidate_1 < fitness_candidate_2: 
            matting_pool.append(candidate_1)
        else:
            matting_pool.append(candidate_2)
    print("matting_pool", matting_pool)
    ###########################################################################################################
    ###########################################################################################################


    new_population= []
    while len(new_population) < n:

        ###########################################################################################################
        ###########################################################################################################
        # Cruzamiento
        print("\nCrossover.......................................................") 
        r = random.random()
        father_index = matting_pool[random.randint(0, n - 1)]
        mother_index = matting_pool[random.randint(0, n - 1)]
        father = population[ father_index, 0:population.shape[1]-1 ]
        mother = population[ mother_index, 0:population.shape[1]-1 ]
        print("father:", father, " mother:", mother, " crossover_uniform")
        if r <= prop_crossover:
            
            betha1 = np.random.uniform(-BLX_alpha, 1 + BLX_alpha)     
            betha2 = np.random.uniform(-BLX_alpha, 1 + BLX_alpha)     
            print("yes crossover...", " betha1:", betha1, " betha2:", betha2)   
            c1 = father[0,0] + betha1*( mother[0,0] - father[0,0] )
            c2 = father[0,1] + betha2*( mother[0,1] - father[0,1] )
            child = [[c1, c2]]
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
            child[0,pos] = new_gene
            print("yes mutation child: ", child, " pos:", pos) # mutation bit flip
        else:
          print("not mutation")
        ###########################################################################################################
        ###########################################################################################################

        new_population.append(child[0].tolist())




    new_population = np.array(new_population)
    print("\nnew_population:")
    print(new_population, new_population.shape)
    population = compute_fitness(new_population)



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


#########################################################################################33
# QUESTIONS : 
# cuando no hay crossover, que padre tomamos para darle la inf al hijo
# esta bien la gráfica de la funcion presentada en el informe?
