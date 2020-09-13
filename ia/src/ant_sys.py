import numpy as np
import random
#random.seed(1)
from numpy.random import seed
import pandas as pd
import sys
#seed(1)
np.set_printoptions(precision=4, suppress=True)


def compute_fitness(population, DIST):   
    for chromosome in population: 
      total_dist = 0
      for i in range(chromosome.shape[0] - 2):
        #print(chromosome[i], " , ", chromosome[i+1], " = ", DIST[chromosome[i], chromosome[i+1]])
        total_dist += DIST[chromosome[i], chromosome[i+1]]
      chromosome[-1] = total_dist
  
def print_population(pop):
    pop_tmp = pop.copy().astype(str)
    #for row in pop_tmp:
    #    for col in row:


    pop_tmp[pop_tmp == "0"] = 'A'
    pop_tmp[pop_tmp == "1"] = 'B'
    pop_tmp[pop_tmp == "2"] = 'C'
    pop_tmp[pop_tmp == "3"] = 'D'
    pop_tmp[pop_tmp == "4"] = 'E'
    pop_tmp[pop_tmp == "5"] = 'F'
    pop_tmp[pop_tmp == "6"] = 'G'
    pop_tmp[pop_tmp == "7"] = 'H'
    pop_tmp[pop_tmp == "8"] = 'I'
    pop_tmp[pop_tmp == "9"] = 'J'

    population_df = pd.DataFrame(data=pop_tmp, columns=["city", "city", "city", "city", "city", "city", "city", "city", "city", "city", "cost"])
    print("\nAnts path:\n", population_df)
   
###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 10
pheromone_factor = 0.1
alpha = 1.0
betha = 1.0
rho = 0.01
Q = 1.0
city_o = 3
iterations = 200
#cities = ['A', 'B', 'C', 'D', 'E']
#cities_indexes = [0,1,2,3,4]
cities = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
cities_indexes = [0,1,2,3,4, 5, 6, 7, 8, 9]
###########################################################################################################
###########################################################################################################

DIST = np.array([ [ 0,  12, 3,  23, 1, 5,  23, 56, 12, 11 ],
                  [ 12, 0,  9,  18, 3,  41, 45, 5, 41, 27 ],
                  [ 3,  9,  0,  89, 56, 21, 12, 48, 14, 29 ],
                  [ 23, 18, 89,  0, 87, 46, 75, 17, 50, 42 ],
                  [ 1,  3,  56, 87,  0, 55, 22, 86, 14, 33 ],
                  [ 5,  41, 21, 46, 55,  0, 21, 76, 54, 81 ],
                  [ 23, 45, 12, 75, 22, 21,  0, 11, 57, 48 ],
                  [ 56, 5,  48, 17, 86, 76, 11, 0,  63, 24 ],
                  [ 12, 41, 14, 50, 14, 54, 57, 63, 0,  9  ],
                  [ 11, 27, 29, 42, 33, 81, 48, 24, 9,  0  ]])
'''
DIST = np.array(  [ 
    [ 0.0,  12.0,   3.0,    23.0,   1.0 ],
    [ 12.0,	0.0,	9.0,	18.0,	3.0 ],
    [ 3.0,	9.0,	0.0,	89.0,	56.0 ],
    [ 23.0,	18.0,	89.0,   0.0,	87.0 ],
    [ 1.0,	3.0,	56.0,	87.0,	0.0 ]    
 ] )
'''
VISIBILITY = 1/DIST
VISIBILITY[np.diag_indices_from(VISIBILITY)] = 0.0
PHEROMONE = np.zeros( (DIST.shape[0], DIST.shape[1]) ) + 0.1

print("parameters.....................................................")
print("population_size:", population_size)
print("pheromone_factor:", pheromone_factor)
print("alfa:", alpha)
print("betha:", betha)
print("rho:", rho)
print("Q:", Q)
print("city origin:", cities[city_o])
print("iterations:", iterations)
print("...............................................................\n")

DIST_df = pd.DataFrame(data=DIST, columns=cities, index=cities)
VISIBILITY_df = pd.DataFrame(data=VISIBILITY, columns=cities, index=cities)
PHEROMONE_df = pd.DataFrame(data=PHEROMONE, columns=cities, index=cities)
print("Distances:\n", DIST_df, "\n\n")
print("Visibility:\n", VISIBILITY_df, "\n\n")
print("Pheromone:\n", PHEROMONE_df, "\n\n")



fitness_history = []

for iter in range(iterations):
    print("\n\nITERATION ", iter , "...........................................................................")
    ###########################################################################################################
    ###########################################################################################################
    population = np.zeros( (population_size, len(cities) + 1) ).astype(int)
    
    for ant_index in range(population_size):
        print("\nAnt ", ant_index, "...........................................................................")
        # compute probability of next city
        temp_cities = cities_indexes.copy()
        city_from = city_o
        temp_cities.remove(city_from)
        # con este ciclo, obtenemos el camino de la horamiga, en cada iteracion avanxzamos en una ciudad

        path = [ cities[city_o]]
        path_indexes = [city_o]

        while len(temp_cities) > 0:
            print("Current city: ", cities[city_from])    
            acc = 0
            temp_prop = [] # save the numerator of equation pag 28 (Ant system slides)
            for city_to in temp_cities:
                r_ij =  PHEROMONE[city_from, city_to]
                n_ij =  VISIBILITY[city_from, city_to]
                temp_prop.append( ( r_ij**alpha ) * ( n_ij**betha ) )
                acc += temp_prop[-1]
                
                #print(cities[city_from], "->", cities[city_to], "r", temp_prop[-1])
                print("r_"+cities[city_from]+"_"+cities[city_to], "=", r_ij, " | n_"+cities[city_from]+"_"+cities[city_to], "=", n_ij, " | r*n=", temp_prop[-1])
            
            probabilities = np.hstack(( np.array(temp_cities).reshape(len(temp_cities), 1),   (np.array(temp_prop)/acc).reshape(len(temp_cities), 1) ))
            print("probabilities:\n", probabilities) 
            #print(random.choices(probabilities[:,0], probabilities[:,1]))
            choosen_city = int(random.choices(probabilities[:,0], probabilities[:,1])[0])     
            path_indexes.append( choosen_city )  
            path.append( cities[choosen_city] )

            print("choose city: ", choosen_city, "->", cities[ choosen_city ], "\n")            
            city_from = choosen_city
            temp_cities.remove(city_from)
        
        print("path Ant", ant_index, ":", path_indexes, "->", path)
        population[ant_index, 0:len(cities)] = path_indexes
        compute_fitness(population, DIST)

    print_population(population)
    
    
    # here, we compute the pheromone trail by each ant
    pheromone_trail = np.zeros( (PHEROMONE.shape[0], PHEROMONE.shape[1]) )
    for ant_index in range(population.shape[0]):
        for city in range(population.shape[1]-2):
            pheromone_trail[ population[ant_index, city], population[ant_index, city + 1]] = Q/population[ant_index, -1]
            pheromone_trail[ population[ant_index, city + 1], population[ant_index, city]] = Q/population[ant_index, -1]
            
            #print("cities:", population[ant_index, city], population[ant_index, city + 1], Q/population[ant_index, -1])
            
    #print(pheromone_trail)
    PHEROMONE = PHEROMONE*(1-rho)
    PHEROMONE = PHEROMONE + pheromone_trail

    PHEROMONE_df = pd.DataFrame(data=PHEROMONE, columns=cities, index=cities)
    print("\nNew Pheromone:\n", PHEROMONE_df)

    
###########################################################################################################
###########################################################################################################

print("\n..................................................")
print("\nRESULT..........................................")
print("Population:")
print_population(population)
best = np.argmin(population[:, -1])
print("\nAnt system result:" )
for i in range(population.shape[1]-1):
    print( cities[ population[best,i] ] + '-' , end = '') 

#print( cities[ population[best,0] ]+"-"+cities[ population[best,1] ]+"-"+cities[ population[best,2] ]+"-"+cities[ population[best,3] ]+"-"+cities[ population[best,4]]+'-'+cities[ population[best,5] ]+"-"+cities[ population[best,6] ]+"-"+cities[ population[best,7] ]+"-"+cities[ population[best,8] ]+"-"+cities[ population[best,9] ])



