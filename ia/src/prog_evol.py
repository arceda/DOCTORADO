import numpy as np
import random
#random.seed(1)
from numpy.random import seed
import pandas as pd
import sys

import graphviz
from graphviz import Digraph
from graphviz import Source
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


# attr_state. cantidad de atributos que definen un automata = 7
# automata -> vector que representa el automata, cada 7 atributos es un estado
def draw_automata(num_attr_state, automata, states, file_name = 'automata'):
    dot = "digraph G { \n"
    dot += "graph [ bgcolor=white, resolution=150, fontname=Arial, fontcolor=blue, fontsize=8 ]; \n"
    dot += "node [ fontname=Arial, fontcolor=blue, fontsize=8]; \n"
    dot += "edge [ fontname=Arial, fontcolor=red, fontsize=8 ]; \n"
    dot += "rankdir=LR; \n"
    dot += 'size="8,5" \n\n'
    dot += "node [shape = point ]; q_o; \n"
    dot += "node [shape = circle ]; \n"    
        
    for i in range( int(automata.shape[0]/num_attr_state) ):
        current_state = states[ i ]  
        
        label_1 = automata[i*num_attr_state + 1] + '|' + automata[i*num_attr_state + 3]
        label_2 = automata[i*num_attr_state + 2] + '|' + automata[i*num_attr_state + 4]

        dot += current_state + "->" + automata[i*num_attr_state + 5] + ' [ label = "' + label_1 + '" ]' + ';\n'
        dot += current_state + "->" + automata[i*num_attr_state + 6] + ' [ label = "' + label_2 + '" ]' + ';\n'

        if automata[i*num_attr_state] == '2':
            dot += "q_o ->" + current_state + ';\n'
        
    dot += "}"
    #print(dot)

    
    graph = graphviz.Source(dot, format='png') # si no ponemos format, por defecto es pdf
    graph.render(file_name,view=True)


# esta funcion, corrige el automata:
# que solo tenga una soluicion incial
# si hay estados nulos, nadie debe apuntar a ellos
def automata_correction(num_attr_state, automata, states):
    # estado inincial, escogemos solo uno de los estados como estado inicial
    pos = random.randint(0, num_states-1)
    automata[pos*num_attr_state] = '2'

    # revisamos si hay estados no activos
    active_states = []
    inactive_states = []
    # por cada estado
    for i in range( int(automata.shape[0]/num_attr_state) ):
        #print(automata[i*num_attr_state])
        if automata[i*num_attr_state] == '0':
            inactive_states.append(states[i])
        if automata[i*num_attr_state] == '1':
            active_states.append(states[i])

    #print(active_states)
    #print(inactive_states)
    print("Inactive states: ", inactive_states, "automata:", automata)

    # si no hay ninguna estado activo, el automata no es  factible
    if len(active_states) == 0:
        return False
    else:
        for i in range( int(automata.shape[0]/num_attr_state) ):
            #print(automata[i*num_attr_state + 5], inactive_states)
            if automata[i*num_attr_state + 5] in inactive_states:
                pos = random.randint(0, len(active_states)-1)
                automata[i*num_attr_state + 5] = active_states[pos]

            if automata[i*num_attr_state + 6] in inactive_states:
                pos = random.randint(0, len(active_states)-1)
                automata[i*num_attr_state + 6] = active_states[pos]
    
    print("corrected automata:", automata)
    return True


def fitness(automata, db, states, num_attr_state):
    initial_state_index = np.where(automata == '2')[0][0]
    current_state_index = initial_state_index
    out_put = []
    states_hist = []
    #states_hist.append(states[int(current_state_index/num_attr_state)])
    #print(initial_state_index, automata)
    for i in range(len(db)):
        
        print("current state:", states[int(current_state_index/num_attr_state)])
        print("current_state_index + 1:", automata[current_state_index + 1])
        print("current_state_index + 5:", automata[current_state_index + 5])
        print("current_state_index + 2:", automata[current_state_index + 2])
        print("current_state_index + 6:", automata[current_state_index + 6])

        if automata[current_state_index + 1] == db[i]:            
            out_put.append( automata[current_state_index + 3] )
            next_state = automata[current_state_index + 5]            
        if automata[current_state_index + 2] == db[i]:            
            out_put.append( automata[current_state_index + 4] )
            next_state = automata[current_state_index + 6]

        print("next_state:", next_state)
        #print(next_state, np.where(np.array(states) == next_state))
        current_state_index = (np.where(np.array(states) == next_state)[0][0])*num_attr_state
        states_hist.append(states[int(current_state_index/num_attr_state)])

    print("automata:", automata)
    print("input:   ", db)
    print("output:  ", out_put)
    print("st_his:  ", states_hist)

###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 8
num_states = 5 # maximo numero de estados
num_attr_state = 7


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
print("iterations:", iterations)
print("replication_prop:", replication_prop)
print("replication_torneo_n:", replication_torneo_n)
print("crossover_prop:", crossover_prop)
print("crossover_torneo_n:", crossover_torneo_n)
print("crossover_point (ramdom point)")
print("mutation_prop:", mutation_prop)
print("mutation_torneo_n:", mutation_torneo_n)
print("...............................................................\n")

'''
digraph G {

rankdir=LR;
size="8,5"

LR_0 -> LR_2 [ label = "SS(B)" ];
LR_0 -> LR_1 [ label = "SS(S)" ];
LR_1 -> LR_3 [ label = "S($end)" ];
LR_2 -> LR_6 [ label = "SS(b)" ];
LR_2 -> LR_5 [ label = "SS(a)" ];
LR_2 -> LR_4 [ label = "S(A)" ];
LR_5 -> LR_7 [ label = "S(b)" ];
LR_5 -> LR_5 [ label = "S(a)" ];
LR_6 -> LR_6 [ label = "S(b)" ];
LR_6 -> LR_5 [ label = "S(a)" ];
LR_7 -> LR_8 [ label = "S(b)" ];
LR_7 -> LR_5 [ label = "S(a)" ];
LR_8 -> LR_6 [ label = "S(b)" ];
LR_8 -> LR_5 [ label = "S(a)" ];

node [shape = doublecircle]; LR_0 LR_3 LR_4 LR_8;
node [shape = circle]; LR_1 LR_2 LR_5 LR_6 LR_7;

node [shape = point ]; q_o;
q_o -> LR_0;

}
'''

states = ['A', 'B', 'C', 'D', 'E']
dataset = ['0', '1', '1', '0', '1', '1', '0', '1', '1', '0', '1', '1', '0', '1', '1']

population = np.zeros(( population_size, num_attr_state*num_states + 1 )).astype(str)

print( "building solutions...\n" )

pop_count = 0
while pop_count < population_size:
    automata = np.zeros( num_attr_state*num_states ).astype(str) #
    for i in range(num_states): # aqui creamos cada estado
        tmp_state = np.zeros(num_attr_state).astype(str)        

        tmp_state[0] = str(random.randint(0, 1)) # 0-> no activo, 1 -> activo, 2 -> inicial
        tmp_state[1] = str(random.randint(0, 1)) # para qu sea AFD, value_1 y value_2, deben ser distintos
        tmp_state[2] = str(1 - int(tmp_state[1]))
        tmp_state[3] = str(random.randint(0, 1)) # imprime esto cuando entra value_1
        tmp_state[4] = str(random.randint(0, 1)) # imprime esto cuando entra value_2
        tmp_state[5] = str(random.choices(  states,  [0.2, 0.2, 0.2, 0.2, 0.2] )[0])
        tmp_state[6] = str(random.choices(  states,  [0.2, 0.2, 0.2, 0.2, 0.2] )[0])

        automata[i*num_attr_state : i*num_attr_state + num_attr_state] = tmp_state

    # draw automata antes de verificar lsoomerrores
    #draw_automata(num_attr_state, automata, states, 'automata1')

    det = automata_correction(num_attr_state, automata, states)    
    if det:

        automata = np.array([
            '0', '0', '1', '0', '0', 'E', 'D', 
            '0', '1', '0', '1', '1', 'E', 'D', 
            '0', '0', '1', '1', '0', 'E', 'E', 
            '2', '1', '0', '1', '1', 'D', 'E', 
            '1', '1', '0', '1', '1', 'E', 'D'])
        print(automata)

        draw_automata(num_attr_state, automata, states, 'automata')
        population[pop_count, 0:num_attr_state*num_states] = automata
        population[pop_count, num_attr_state*num_states] = fitness(automata, dataset, states, num_attr_state)
        #print("automata valid: ", automata)
        

    #else:
        #print("automata no valid: ", automata)

    
    

    sys.exit(0)

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

