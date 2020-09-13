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

def oper(oper1, oper2, func):
    #print(oper1, oper2, func)
    if func == "*":
        return True, oper1*oper2
    if func == "/":
        if oper2 == 0:
            return False, None
        else:
            return True, oper1/oper2
    if func == "+":
        return True, oper1+oper2
    if func == "-":
        return True, oper1-oper2

    #print(population)
def check_solution_feasibility(ps, db): #ps: possible solution, db: dataset
    #print(ps.shape)
    
    rmse = 0
    cache = []
    for sample in db:
        #print("ps", ps)
        possible_sol = ps.tolist() 
        # estos es muy importante, si lo dejamos como numpy , cuando remplacemos  ps[i] = '0.01'
        # solo le asignara el primer caracter, osea al final quedara ps[i] = '0'
        
        #print("sample", sample, "possible_sol", possible_sol)
        #ps[ps == "x"] = sample[0]
        #print(ps)
        for i  in range(ps.shape[0]):
            if possible_sol[i] == "x":
                possible_sol[i]  = str(sample[0])
                #print("str(sample[0])", str(sample[0]))
                #print("remplazing x by ", sample[0], sample, possible_sol[i])
        #print(ps, "\n")
        #print(possible_sol)
        det_1, temp_1 = oper(float(possible_sol[0]), float(possible_sol[2]), possible_sol[1])
        det_2, temp_2 = oper(float(possible_sol[4]), float(possible_sol[6]), possible_sol[5])
        if det_1 == False or det_2 == False:
            return False, None, None
        
        #print(possible_sol, sample, temp_1, temp_2)
        det, result = oper( temp_1, temp_2, ps[3] )         
        if det == False:
            return False, None, None
        
        temp = (result - sample[1])**2 
        cache.append( [ result, temp ] )
        rmse += temp

    #return True, rmse**0.5/dataset.shape[0], np.array(cache)
    return True, rmse/dataset.shape[0], np.array(cache)
   
###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 8
num_genes = 7 # tenemos numeros de 0 a 511 => necesitamos binarios de 10 bits
iterations = 1000
replication_prop = 0.2
replication_torneo_n = 3
crossover_prop = 0.4
crossover_torneo_n = 2
mutation_prop = 0.4
mutation_torneo_n = 3


###########################################################################################################
###########################################################################################################

print("parameters.....................................................")
print("population_size:", population_size)
print("num_genes:", num_genes)
print("iterations:", iterations)
print("replication_prop:", replication_prop)
print("replication_torneo_n:", replication_torneo_n)
print("crossover_prop:", crossover_prop)
print("crossover_torneo_n:", crossover_torneo_n)
print("crossover_point (ramdom point)")
print("mutation_prop:", mutation_prop)
print("mutation_torneo_n:", mutation_torneo_n)
print("...............................................................\n")

consts = np.array(["-5", "-4", "-3", "-2", "-1", "1", "2", "3", "4", "5", "x", "x", "x", "x", "x", "x", "x", "x"])
functions = np.array([ "+", "-", "*", "/" ])

dataset = np.array([ [0, 0], [0.1, 0.005], [0.2, 0.02], [0.3, 0.045], [0.4, 0.08], 
                    [0.5, 0.125], [0.6, 0.18], [0.7, 0.245], [0.8, 0.32], [0.9, 0.405]])

population = np.zeros(( population_size, num_genes + 1 ))
population = population.astype(str)

print( "building solutions...\n" )

pop_count = 0
while pop_count < population_size:
    np.random.shuffle(consts)
    np.random.shuffle(functions)
    possible_solution = np.array([ consts[0],  functions[0], consts[1],  functions[1], consts[2],  functions[2], consts[3]])

    #print("possible_solution", possible_solution)

    det, fitness, cache = check_solution_feasibility(possible_solution, dataset)
    if det:
        population[pop_count, 0:num_genes] = possible_solution
        population[pop_count, num_genes] = fitness
        #print("population", population[pop_count])

        temp = np.zeros((cache.shape[0], 3))
        temp[:,0] = dataset[:, 1]
        temp[:,1] = cache[:, 0]
        temp[:,2] = cache[:, 1]
        #print( "Solution:",  possible_solution )
        #print( pd.DataFrame(data=temp, columns=["y", "y'", "(y-y')^2"]) )
        #print( "fitness:",  fitness, "\n" )
        pop_count += 1

print(population)

population_df = pd.DataFrame(data=population, columns=["oper_1", "func", "oper_2", "func", "oper_3", "func", "oper_4", "fitness"])
print("Initial population:\n", population_df)

#sys.exit(0)

fitness_history = []

for iter in range(iterations):
    print("\n\nITERATION ", iter , "...........................................................................")
    ###########################################################################################################
    ###########################################################################################################
    print(population_df)    
    #new_population = np.zeros( (population_size, num_genes + 1) ).astype(str)
    new_population = []
    
    fitness_history.append(  [  np.mean(population[:, num_genes].astype(float) ), np.min(population[:, num_genes].astype(float) ) ] )

    while len(new_population) < population_size:

        ###########################################################################################################
        ###########################################################################################################
        # Replication
        r = random.random()
        if r <= replication_prop: 
            print("\nReplication..............................................................")
            sample_replication = population[torneo(population, replication_torneo_n)].copy()
            new_population.append( sample_replication )
            print("new chormosome added -> ", sample_replication)
            print("new_pop.size=", len(new_population)) 

        if len(new_population) >= population_size: break
        ###########################################################################################################
        ###########################################################################################################
        # Crossover
        r = random.random()
        if r <= crossover_prop: 
            print("\nCrossover................................................................")
            father = population[torneo(population, crossover_torneo_n)]
            mother = population[torneo(population, crossover_torneo_n)]

            print("father:", father, " mother:", mother)    

            pos = random.randint(0, num_genes - 1)
            
            child_1 = np.zeros( num_genes + 1).astype(str) 
            child_2 = np.zeros( num_genes + 1).astype(str) 

            child_1[0:pos] = father[0:pos]            
            child_1[pos:num_genes] = mother[pos:num_genes]

            child_2[0:pos] = mother[0:pos]
            child_2[pos:num_genes] = father[pos:num_genes]

            #print("position:", pos, " Child_1:", child_1, " child_2:", child_2) 

            det_1, fitness_1, cache_1 = check_solution_feasibility(child_1, dataset)                
            det_2, fitness_2, cache_2 = check_solution_feasibility(child_2, dataset)
            
            #if det_1 == True:
            if det_1 == True and fitness_1 < float(child_1[num_genes]):
                child_1[num_genes] = fitness_1
                new_population.append( child_1 )
                print("position:", pos, "Child_1 added", child_1, " FEASIBLE")  
                print("new_pop.size=", len(new_population))  
                if len(new_population) >= population_size: break
            else:
                print("position:", pos, "Child_1 NOT added", child_1, " NOT FEASIBLE")  
            #if det_2 == True:
            if det_2 == True and fitness_2 < float(child_2[num_genes]):
                child_2[num_genes] = fitness_2
                new_population.append( child_2 )
                print("position:", pos, "Child_2 added", child_2, " FEASIBLE") 
                print("new_pop.size=", len(new_population))   
                if len(new_population) >= population_size: break
            else:
                print("position:", pos, "Child_2 NOT added", child_2, " NOT FEASIBLE")  
        ###########################################################################################################
        ###########################################################################################################
        # Mutacion    
        r = random.random()
        
        if r <= mutation_prop:  
            print("\nMutation................................................................")  
            sample_replication = population[torneo(population, mutation_torneo_n)].copy()
            # selccionamos el terminal o funcion a mutar en ambos hijos
            pos = random.randint(0, num_genes - 1)
                        
            print("chromosome original:", sample_replication)

            if pos % 2 == 0: # replace by const
                while sample_replication[pos] == consts[0]: # con esto aseguramos de no mutar por el mismo gen
                    np.random.shuffle(consts) 
                sample_replication[pos] = consts[0]
            else: # replace by function
                while sample_replication[pos] == functions[0]:
                    np.random.shuffle(functions) 
                sample_replication[pos] = functions[0]
            
            det_1, fitness_1, cache_1 = check_solution_feasibility(sample_replication, dataset)  
            if det_1 and fitness_1 < float(sample_replication[num_genes]):
            #if det_1:
                new_population.append( sample_replication )   
                sample_replication[num_genes] = fitness_1
                print("chromosome mutatted added:", sample_replication, " at pos: ", pos)                
            else:
                print("chromosome mutatted NOT added:", sample_replication, " at pos: ", pos, " NOT FEASIBLE ")
            print("new_pop.size=", len(new_population))
           

    population = np.array(new_population)
    population_df = pd.DataFrame(data=population, columns=["oper_1", "func", "oper_2", "func", "oper_3", "func", "oper_4", "fitness"])


###########################################################################################################
###########################################################################################################

print("\n..................................................")
print("\nRESULT..........................................")
print("Population:")
print(population_df)
print("Genetic algorithm result:",  np.min(population[:, num_genes].astype(float)))


ps = population[ np.argmin(population[:, num_genes].astype(float)) ]
print("BEST SOLUTION:\n")
print( '|', ps[0], '|', ps[1], '|', ps[2], '|', ps[3], '|', ps[4], '|', ps[5], '|', ps[6], '|')

print('digraph G{')
for i in range(ps.shape[0] - 1):
  print( i,  '[ label="', ps[i],'" ];')
print("1 -> 0; \n 1 -> 2;  \n 5 -> 4;  \n 5 -> 6;  \n 3 -> 1;  \n 3 -> 5;")
print('}')


import matplotlib.pyplot  as plt

fitness_history = np.array(fitness_history)

fig = plt.figure()
ax = plt.axes()

ax.plot(range(fitness_history.shape[0]), fitness_history[:, 0], label="pop_fitness mean")
ax.plot(range(fitness_history.shape[0]), fitness_history[:, 1], label="pop_fitness min")
ax.legend()
ax.set_ylim([0,1])
plt.xlabel('Iterations')
plt.ylabel('Fitness')

plt.savefig('fitness_history.png', dpi = 300)

