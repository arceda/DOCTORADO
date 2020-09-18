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
    dot += "graph [ bgcolor=white, resolution=400, fontname=Arial, fontcolor=blue, fontsize=8 ]; \n"
    dot += "node [ fontname=Arial, fontcolor=blue, fontsize=8]; \n"
    dot += "edge [ fontname=Arial, fontcolor=red, fontsize=8 ]; \n"
    dot += "rankdir=LR; \n"
    dot += 'size="8,5" \n\n'
    dot += "node [shape = point ]; q_o; \n"
    dot += "node [shape = circle ]; \n"    
        
    for i in range( int(automata.shape[0]/num_attr_state) ):
        current_state = states[ i ]  
        
        if automata[i*num_attr_state] == '0':
            continue

        label_1 = automata[i*num_attr_state + 1] + '|' + automata[i*num_attr_state + 3]
        label_2 = automata[i*num_attr_state + 2] + '|' + automata[i*num_attr_state + 4]

        dot += current_state + "->" + automata[i*num_attr_state + 5] + ' [ label = "' + label_1 + '" ]' + ';\n'
        dot += current_state + "->" + automata[i*num_attr_state + 6] + ' [ label = "' + label_2 + '" ]' + ';\n'

        if automata[i*num_attr_state] == '2':
            dot += "q_o ->" + current_state + ';\n'
        
    dot += "}"
    #print(dot)
    #node [shape = doublecircle]; LR_0 LR_3 LR_4 LR_8;
    #node [shape = circle]; LR_1 LR_2 LR_5 LR_6 LR_7;
    
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
    #print("Inactive states: ", inactive_states, "automata:", ''.join(automata.tolist()) )

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
    
    #print("corrected automata:", ''.join(automata.tolist()) )
    return True


def fitness(automata, db, states, num_attr_state):
    initial_state_index = np.where(automata == '2')[0][0]
    current_state_index = initial_state_index
    out_put = []
    states_hist = []
    #states_hist.append(states[int(current_state_index/num_attr_state)])
    #print(initial_state_index, automata)
    acc_count = 0
    for i in range(len(db)):        
        #print("current state:", states[int(current_state_index/num_attr_state)])
        #print("current_state_index + 1:", automata[current_state_index + 1])
        #print("current_state_index + 5:", automata[current_state_index + 5])
        #print("current_state_index + 2:", automata[current_state_index + 2])
        #print("current_state_index + 6:", automata[current_state_index + 6])

        
        if automata[current_state_index + 1] == db[i]:    
            out_ =  automata[current_state_index + 3]
            out_put.append( automata[current_state_index + 3] )
            next_state = automata[current_state_index + 5]            
        if automata[current_state_index + 2] == db[i]:  
            out_ =  automata[current_state_index + 3]          
            out_put.append( automata[current_state_index + 4] )
            next_state = automata[current_state_index + 6]

        # para obtener el acc
        if db[i] == out_:
            acc_count += 1

        #print("next_state:", next_state)
        #print(next_state, np.where(np.array(states) == next_state))
        current_state_index = (np.where(np.array(states) == next_state)[0][0])*num_attr_state
        states_hist.append(states[int(current_state_index/num_attr_state)])

    #print("automata:", automata)
    #print("input:   ", db)
    #print("output:  ", out_put)
    #print("st_his:  ", states_hist)
    return float(acc_count/len(db)), out_put, states_hist

def mutate(automata, num_attr_state, num_states, states):
    automata_mutated = automata.copy()
    active_states = []
    inactive_states = []
    # por cada estado
    for i in range( int(automata.shape[0]/num_attr_state) ):
        #print(automata[i*num_attr_state])
        if automata[i*num_attr_state] == '0':
            inactive_states.append(states[i])
        if automata[i*num_attr_state] == '1':
            active_states.append(states[i])
    r = random.random()
    if 0.0 <= r <= 0.1:        
        pos = random.randint(0, len(states) - 1) # para saber que stado moutamos
        print ("mututation = desactivar un estado", "- state pos:", pos)
        automata_mutated[ pos*num_attr_state ] = '0'
        det = automata_correction(num_attr_state, automata_mutated, states)
        if det == False:
            return False, None

    if 0.1 < r <= 0.3:        
        pos = random.randint(0, len(states)  - 1) # para saber que stado moutamos
        print ("mutation = cambiar estado incial", "- state pos:", pos)
        initial_state_index = np.where(automata_mutated == '2')[0][0]    
        automata_mutated[initial_state_index] = '1'
        automata_mutated[ pos*num_attr_state ] = '2'

    if 0.3 < r <= 0.5:
        pos = random.randint(0, len(states)  - 1) # para saber que stado moutamos
        print ("mutation = cambiar simbolo de entrada", "- state pos:", pos)
        automata_mutated[pos*num_attr_state + 1] = str(1 - int( automata_mutated[pos*num_attr_state + 1] ))
        automata_mutated[pos*num_attr_state + 2] = str(1 - int( automata_mutated[pos*num_attr_state + 1] ))

    if 0.5 < r <= 0.7:
        pos = random.randint(0, len(states)  - 1) # para saber que stado moutamos
        pos_2 = random.randint(0, 1)  # para saber que salida mutamos
        print ("mutation = cambiar un simbolo de salida", "- state pos:", pos, " salida:", pos_2)
        automata_mutated[pos*num_attr_state + 3 + pos_2] = str(1 - int( automata_mutated[pos*num_attr_state + 3 + pos_2] ))

    if 0.7 < r <= 0.9:
        pos = random.randint(0, len(states)  - 1) # para saber que stado moutamos        
        pos_2 = random.randint(0, len(active_states)  - 1)  # por este estado se cambiara el estado de salida
        pos_3 = random.randint(0, 1)  # se cambiara la salida 1 o 2
        print ("mutation = cambiar un estado de salida", "- state pos:", pos, 'por el estado:', active_states[pos_2], 'la salida:', pos_3 )
        automata_mutated[pos*num_attr_state + 5 + pos_3] = active_states[pos_2]

    if 0.9 < r <= 1.0: 
        if  len(inactive_states) > 0:  
            inactive_sta = inactive_states[random.randint(0, len(inactive_states)  - 1)] # estado inactivo a activar
            #print(inactive_sta, states)
            state_index = np.where(np.array(states) == inactive_sta)[0][0] #ubicamos el estado inactivo en el automata
            print ("mutation = activar un estado", "- iinactive_sta to active:", inactive_sta) # para saber que stado moutamos
            automata_mutated[state_index*num_attr_state] = '1'
        else:
            return False, None

    return True, automata_mutated

def show_pop(population, dataset):
    for chromosome in population:
        automata = chromosome[0:num_attr_state*num_states]
        out_put = chromosome[num_attr_state*num_states]
        fit = chromosome[num_attr_state*num_states + 1]

        for i in range(len(states)):
            automata_state = automata[i*num_attr_state:i*num_attr_state + num_attr_state]
            print( ''.join(automata_state.tolist()) + '-', end='')
        #print( ''.join(automata.tolist()), '--' , ''.join(dataset), '-', out_put, '-', fit )
        print( '--' , ''.join(dataset), '--', out_put, '--', fit )
    print()

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

states = ['A', 'B', 'C', 'D', 'E']
dataset = ['0', '1', '1', '0', '1', '1', '0', '1', '1', '0', '1', '1', '0', '1', '1']


population = np.zeros(( population_size, num_attr_state*num_states + 2 )).astype(str)

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
        '''
        automata = np.array([
            '0', '0', '1', '0', '0', 'E', 'D', 
            '0', '1', '0', '1', '1', 'E', 'D', 
            '0', '0', '1', '1', '0', 'E', 'E', 
            '2', '1', '0', '1', '1', 'D', 'E', 
            '1', '1', '0', '1', '1', 'E', 'D'])
        print(automata)
        '''
        #draw_automata(num_attr_state, automata, states, 'automata')
        fit, out_put, states_hist = fitness(automata, dataset, states, num_attr_state)
        population[pop_count, 0:num_attr_state*num_states] = automata
        population[pop_count, num_attr_state*num_states] = ''.join(out_put)
        population[pop_count, num_attr_state*num_states + 1] = fit      

        pop_count += 1

show_pop(population, dataset)

fitness_history = []

for iter in range(iterations):
    print("\n\nITERATION ", iter , "...........................................................................")
    ###########################################################################################################
    ###########################################################################################################
    print("Current population:")
    show_pop(population, dataset)

    offsprings = np.zeros(( population_size, num_attr_state*num_states + 2 )).astype(str)
        
    ##############################################################################################################
    # MUTATION

    for i in range(population.shape[0]):
        automata = population[i, 0:num_attr_state*num_states]

        det, automata_mut = mutate(automata, num_attr_state, num_states, states)

        if det:
            print("automata:", ''.join(automata.tolist()), "automata mutated:", ''.join(automata_mut.tolist()), '\n')
            fit, out_put, states_hist = fitness(automata_mut, dataset, states, num_attr_state)
            offsprings[i, 0:num_attr_state*num_states] = automata
            offsprings[i, num_attr_state*num_states] = ''.join(out_put)
            offsprings[i, num_attr_state*num_states + 1] = fit   
        else:
            print("automata mutated is not valid")

    show_pop(offsprings, dataset)   
    ##############################################################################################################

    sys.exit(0)

   

    population = np.array(new_population)
    population_df = pd.DataFrame(data=population, columns=["oper_1", "func", "oper_2", "func", "oper_3", "func", "oper_4", "fitness"])


###########################################################################################################
###########################################################################################################

print("\n..................................................")
print("\nRESULT..........................................")
print("Population:")
print(population_df)
print("Genetic algorithm result:",  np.min(population[:, num_genes].astype(float)))



