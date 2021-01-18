import numpy as np
import random

simulations = 30

def split(word): 
    return [char for char in word]  

# compute overlap complexity of two space seeds
def slowOC(s_1, s_2):
    s_1 = split(s_1)
    s_2 = split(s_2)

    t_1 = np.zeros( len(s_1) + 2*len(s_2) - 2)
    t_1[len(s_2) - 1:len(s_2) - 1 + len(s_1)] = s_1

    sigma = []
    result = []

    for i in range( len(s_1) + len(s_2) - 1 ):        
        t_2 = np.zeros( len(s_1) + 2*len(s_2) - 2)       
        t_2[i:len(s_2) + i] = s_2

        matches = np.logical_and(t_1, t_2)
        sigma_tmp = (matches == True).sum()
        sigma.append( sigma_tmp )
        result.append( 2**sigma_tmp )     

    return np.sum( result ), sigma


def quicklyOC(s_1, s_2):
    wight_1 = np.where( np.array(s_1) == 1)
    wight_2 = np.where( np.array(s_2) == 1)
    sigma = np.zeros( ( len(s_1) +  len(s_2) - 1 ) )
    #print(sigma)

    print(wight_1, wight_2)
    for i in wight_1[0]:
        for j in wight_2[0]:
            print(i, j, i + j)
            sigma[ i + j ] += 1

    #print(sigma) 


def overlap_complexity(S):
    #quicklyOC( S[0], S[1] )
    total_oc = 0 # sum overlap complexity
    for i in range ( len(S) - 1):
        oc, sigma =  slowOC( S[i], S[i + 1] )
        total_oc += oc

    return total_oc
    #print(total_oc)
    
def count_occurrence_in_seed(S, model):    
    acc = 0
    for seed in S:        
        acc += seed.count(model)
        #print(S_str, model_str, S_str.count(model_str))
    return acc

# find the sibstring in the table of frequencies
def find_position(table, substring):
    if substring == '0011' or substring == '1011' or substring == '0111':
        return 0
    elif substring == '0101' or substring == '1101' or substring == '0111':
        return 1
    elif substring == '0110' or substring == '1110' or substring == '0111':
        return 1
    elif substring == '0101' or substring == '1101' or substring == '0111':
        return 1
    elif substring == '0101' or substring == '1101' or substring == '0111':
        return 1
    elif substring == '0101' or substring == '1101' or substring == '0111':
        return 1

def build_frequency_pattern_table(S):    
    table = []
    patterns = []
    patterns.append( ['0011','1011','0111'] ) #0
    patterns.append( ['0101','1101','0111'] ) #1
    patterns.append( ['0110','1110','0111'] ) #2
    patterns.append( ['1001','1101','1011'] ) #3
    patterns.append( ['1010','1110','1011'] ) #4
    patterns.append( ['1100','1101','1110'] ) #5
    patterns.append( ['0111'] ) #6
    patterns.append( ['1011'] ) #7
    patterns.append( ['1101'] ) #8
    patterns.append( ['1110'] ) #9

    for patt in patterns:
        occurrence = 0
        for model in patt:
            occurrence += count_occurrence_in_seed(S, model)

        table.append(occurrence)

    return table, patterns

#S = [ "1011", "100101" ]
#print(overlap_complexity(S))

#S = ["1011010110101", "1110111000101", "1001110101101"]
#table = build_frequency_pattern_table(S)
#print(table)

################################################################################
# PARAMETERS
################################################################################
#penguins_number = 25
penguins_number = 5
groups_number = 10
iterations = 20
oxigen_reserve = 1

seed_length = 11
seed_weight = 7

################################################################################


################################################################################
# CREATE POPULATION
################################################################################
population = []
groups = []

# create random population of penguin (seed)
for i in range(penguins_number*groups_number):
    penguin = np.zeros( seed_length ).astype(int)
    indexes = list(range( seed_length ))
    random.shuffle(indexes)
    
    for idx in range(seed_weight):
        penguin[ indexes[idx] ] = 1

    penguin = ''.join( list( map(str, penguin) ) )
    population.append( penguin )

population = np.array( population )
oxigen = np.zeros( (population.shape[0], 1) ).astype(int) + 2
population = np.hstack( (population.reshape(  population.shape[0],1 ), oxigen ) )

#print(population)

for i in range( groups_number ):
    groups.append( population[ i*penguins_number : (i+1)*penguins_number ] )

# create the frequency table
table_frequency, patterns = build_frequency_pattern_table( population[:, 0] )
min_pattern = np.argmin(table_frequency)
#print( patterns[min_pattern][0] )

#print( population )

#for gr in groups:
#    print(gr)

################################################################################


################################################################################
# PENGUIN SEARCH
################################################################################
best_oc = 0

for iter in range(iterations):
    print("\n\nITERATION: ", iter, "***************************************")

    #print(groups[0])   
    #print(population)   

    oc_tmp = []
    for gr in groups:
        oc_tmp.append( overlap_complexity( gr[:, 0] ) )
    
    best_oc = np.min( oc_tmp )
    print( "OC of each group:", oc_tmp )
    
    for gr in groups:
        #print("\nGROUP ********************************")
        table_frequency, patterns = build_frequency_pattern_table( gr[:, 0] )
        min_pattern = np.argmin(table_frequency)

        for penguin in gr:
            #print("\nPENGUIN:", penguin[0])
            seed = penguin[0]            
            while int(penguin[1]) > 0:
                #improve seed             
                pos = random.randint( 0, len(seed)-5 ) # -5 porque extraeremos 4 substring
                
                # reemplazamos en el seed por el patron con menor frecuencia en S            
                old_seed = seed         
                seed = seed[0:pos] + patterns[min_pattern][0] + seed[pos + 4:len(seed)]
                #print("seed: ", old_seed, "pos:", pos, "new seed:", seed)

                penguin[0] = seed
                penguin[1] =  str(int(penguin[1]) - 1)
            
    
         
    



