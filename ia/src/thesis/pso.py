import numpy as np
from Bio import SeqIO
import math
import random
import scipy.stats
import pandas as pd
import sys
import numpy as np
from Bio import SeqIO
import re
import os
import glob
np.set_printoptions(precision=4, suppress=True)
random.seed(1)
from numpy.random import seed
seed(1)

######## global variables ##################################### 
current_dir = os.path.dirname(os.path.abspath(__file__))

# particle format
# primero obtenemos la longitud de la secuecnia mas grande ( L )
# luego |x| = K*L


# se usara el algoritmo de Multiple sequence alignment using swarm intelligence
# la secuencia alineada puede tener ocmo minimo = max_lenght_seq y como maximo = 2*max_lenght_seq
# se genera un numero aleatorio que seran la cantidad de gaps para la secuencia ,mas grande, siempre y cuando cumpla con la instruccion anterior
# luego a las demas secuecnias se agragan gaps, de manera tal que tengan la misma longitud que la secuencia mas grande (instrucciuo anterior)

# el acercamiento es un crossover.

def split(word): 
    return [char for char in word]  


def read_sequences(path):
    files = glob.glob(path + "*.fasta")
    max_length = 0
    min_length = 1000000
    sequences = []
    for file in files:
        seqs = SeqIO.parse(file, "fasta")

        for record in seqs:
            seq_str = str(record.seq.upper())
            sequences.append( [ record.id, seq_str ] )
            if max_length < len( seq_str ):                
                max_length = len( seq_str )
            if min_length > len( seq_str ):
                min_length = len( seq_str )

    sequences = np.array( sequences )
    k = sequences.shape[0]
    return sequences, k, max_length, min_length

def read_sequences_s8():
    max_length = 10
    min_length = 7
    sequences = [ 
        ["s1", "ATGCAAG"], 
        ["s2", "TAAGTCAAGT"], 
        ["s3", "ATGCAACT"], 
        ["s4", "TAAGTCATA"],
        ["s5", "ATGGATTC"]
    ]
    
    sequences = np.array( sequences )
    k = sequences.shape[0]
    return sequences, k, max_length, min_length

def build_population(sequences, k, max_length, min_length, population_size):
    population = []
    for index in range(population_size):   
        # cada particula alinead tendra un diferente tamanio 
        align_seqs_length = random.randint(max_length, int(1.5*max_length) )
        # tods con el mismo tamanio
        align_seqs_length = int(1.2*max_length) 

        #print(align_seqs_length)
        total_gaps = []
        total_aligns_seqs = [None]
        for seq in sequences:
            seq_len = len(seq[1])        
            gaps = []
            for i in range( align_seqs_length - seq_len ):
                gaps.append(random.randint(0, seq_len))
            total_gaps.append( gaps )

            # insertamos los gap en la sequencia
            gaps.sort()
            align_seq = split(seq[1]) 
            for gap in gaps:
                align_seq.insert( gap, '-' ) 
            ####################################### 

            total_aligns_seqs.append( ''.join(align_seq) )
        #population.append( total_gaps )
        population.append( total_aligns_seqs )

    population = np.array( population ).astype(object)  
    return population
    #print(total_gaps)

# evalua el score cuando las particulas representan los gaps
def evaluate_solution_gaps(particle, sequences):
    #print(particle, sequences)
    total_seqs = []
    for i in range( len(particle) ):
        #print( particle[i], sequences[i] )
        gaps = particle[i].copy()
        gaps.sort()
        align_seq = split(sequences[i][1])
        #print(gaps, align_seq)
        for gap in gaps:
            align_seq.insert( gap, '-' )

        total_seqs.append(align_seq)

    total_seqs_align = np.array( total_seqs )

    # compute fitness
    cols = total_seqs_align.T
    score = 0
    for col in cols:      
        #(col)  
        if np.all(col == '-'): # si todos son gaps
            score += 0
            #print("0")
        elif len(np.where(col == '-')[0]) > 0: # si hay un gap
            score -= 1
            #print("-1")
        elif np.all(col == col[0]): # si todos los elementos son iguales
            score += 2
            #print("2")

    return total_seqs_align, score

def evaluate_solution(population):
    for particle in population:
        seqs = particle[1:-1]       

        # cast string to list pra tratarlo como una matriz
        total_seqs_align = []
        for seq in seqs:
            seq_list = split(seq)
            total_seqs_align.append( seq_list )

        total_seqs_align = np.array(total_seqs_align)
        #print(total_seqs_align)
        # compute fitness
        cols = total_seqs_align.T
        score = 0
        for col in cols:      
            #(col)  
            if np.all(col == '-'): # si todos son gaps
                score += 0
                #print("0")
            elif len(np.where(col == '-')[0]) > 0: # si hay un gap
                score -= 1
                #print("-1")
            elif np.all(col == col[0]): # si todos los elementos son iguales
                score += 2
                #print("2")

        particle[0] = score
    
def get_gap_indices(particle_1, particle_2):
    gaps_1 = []
    gaps_2 = []
    for i in range(particle_1.shape[0]):
        # matching gaps / total gaps
        seq1_list = split(particle_1[i])
        seq2_list = split(particle_2[i])

        tmp1 = list(np.where( np.array(seq1_list) == '-' )[0])
        tmp2 = list(np.where( np.array(seq2_list) == '-' )[0])

        gaps_1.append( tmp1 )
        gaps_2.append( tmp2 )

    return gaps_1, gaps_2

def diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))
 
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

def distance(particle_1, particle_2):
    gaps_1, gaps_2 = get_gap_indices(particle_1, particle_2)
    #print(gaps_1)
    #print(gaps_2)

    matching_gaps = 0
    total_gaps = 0
    for i in range(len(gaps_1)):
        total_gaps += len(gaps_1[i]) + len(gaps_2[i])
        
        min_len = min( len(gaps_1[i]), len(gaps_2[i]) )
        for j in range(min_len):
            if gaps_1[i][j] == gaps_2[i][j]:
                matching_gaps += 2

    #print(matching_gaps, total_gaps)
    return (total_gaps - matching_gaps)/total_gaps

def particle_to_matrix(particle): #particle is a list of strings
    matrix = []
    for sol in particle:
        matrix.append( split(sol) )
    return np.array( matrix )

def weight(seq): #seq is a list np
    num_gaps = (seq == '-').sum()
    return seq.shape[0] - num_gaps

def remove_base(seq, n): #n: cantidad de bases a eliminar    
    indices = []        
    bases_removed = 0
    for i in range(seq.shape[0]):
        if seq[i] != '-':
            indices.append(i)
            bases_removed += 1
        
        if bases_removed >= n:
            break
    
    result = np.delete(seq, indices)

    print("removing bases ( n =",n,") original:", seq, "result:", result, "indices:", indices)    
    return result

def add_base(left, right, n):
    bases_added = 0
    indices = [] 
    values = [] 
    for i in reversed(range(left.shape[0])):
        if left[i] != '-':
            indices.append(bases_added)
            values.append( left[i] )
            bases_added += 1
        
        if bases_added >= n:
            break

    result = np.insert(right, indices, values)
    print("adding bases ( n =",n,") original:", right, "result:", result, "indices:", indices, "values:", values)    
    return result

def check_size(particle):
    

def crossover(particle_1, particle_2): # particle_1=leader, particle_2 = particle
    point = math.floor(distance(particle_1, particle_2)*len(particle_1[0]))
    cross_point = 5

    particle_1 = particle_to_matrix(particle_1)
    particle_2 = particle_to_matrix(particle_2)     

    #print(particle_1, '\n\n', particle_2, '\n')
    #particle_2[:, 0:cross_point] = particle_1[:, 0:cross_point]
    #print(particle_1, '\n\n', particle_2, '\n')

    new_particle_2 = []
    for i in range(particle_1.shape[0]):
        print("crossover between:\n", particle_1[i], '\n', particle_2[i])

        lp_1 = particle_1[i][0:cross_point]
        rp_1 = particle_1[i][cross_point:particle_1[i].shape[0]] 

        lp_2 = particle_2[i][0:cross_point]
        rp_2 = particle_2[i][cross_point:particle_1[i].shape[0]] 

        #print(lp_1, rp_1)
        #print(lp_2, rp_2)

        # calculamos el weight de la parte derecha de ambas particulas
        w_1 = weight(rp_1)
        w_2 = weight(rp_2)

        if w_2 > w_1:
            tmp_rp_2 = remove_base(rp_2, w_2 - w_1)
            new_seq = np.hstack( ( lp_1, tmp_rp_2 ) )            

        elif w_2 == w_1:
            new_seq = np.hstack( ( lp_1, rp_2 ) )

        else:
            tmp_rp_2 = add_base(lp_2, rp_2, w_1 - w_2)
            new_seq = np.hstack( ( lp_1, tmp_rp_2 ) )        
        
        print("new_seq", new_seq, '\n')
        new_particle_2.append( new_seq )

    check_size( np.array(new_particle_2) )


#sequences, k, max_length, min_length = read_sequences(current_dir + "/seqs/S7/")
sequences, k, max_length, min_length = read_sequences_s8()
max_gaps_allowed = math.floor(0.3*max_length)
gaps_allowed = np.random.randint(max_gaps_allowed+1, size=sequences.shape[0])

print(sequences)
#print(sequences.shape)
#print(k, max_length, min_length)
#print(max_gaps_allowed, gaps_allowed)

###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 20
att_per_seq = int(max_length*0.2)   # gaps dentro de una particula para cada secuencia
vector_size = k*att_per_seq         # 0.2 porque solo se tiene como maximo un 20% de gaps
phi_1 = 2
phi_2 = 2
iterations = 1

###########################################################################################################
###########################################################################################################
print("\n...............................................................")
print("PARAMETERS:")
print("population_size:", population_size)
print("rand_1 and rand_2 = [0.1]")
print("phi_1 and phi_2 = 2")
print("Iterations:", iterations)
print("...............................................................\n")


population = build_population(sequences, k, max_length, min_length, population_size)
population_df = pd.DataFrame(data=population)
print("\nPopulation:\n", population_df)

#total_seqs_align, score = evaluate_solution( [ [4,5], [4,5,7,8], [1,7,8] ], [['01','WGKVNVD'], ['02','WDKVN'], ['03','SKVGGN']] )
#print(total_seqs_align)
#print("score", score)

evaluate_solution(population)
population_df = pd.DataFrame(data=population)
print("\nPopulation:\n", population_df)


best_global_index = np.argmax(population[:,0])
best_global = population[best_global_index]
#print("\nbest_global:\n", best_global)

dist = distance(np.array(['WGKV--NVD', 'WDKV--N--', 'S-KVGGN--']), np.array(['WGK--VNVD', '-WDK---VN', 'S-KVGG-N-']))
#print(dist)

crossover(np.array(['WGKV--NVD', 'WDKV--N--', 'S-KVGGN--']), np.array(['WGK--VNVD', '-WDK---VN', 'S-KVGG-N-']))
crossover(np.array(['WGK--VNVD', 'WDKV--N--', 'S-KVGGN--']), np.array(['WGKV--NVD', '-WDK---VN', 'S-KVGG-N-']))

for iter in range(iterations):
    print("\n***  ITERATION ", iter, "**************************************************************************")
    print("********************************************************************************************\n")

    for i in range( population_size ):
        print( "\nParticle ", i, "***********************************" )
        
        crossover( best_global[1:-1], population[i, 1:-1] )
        
        sys.exit(0)

        w = random.random()
        rand_1 = random.random()
        rand_2 = random.random()

        print("w:", w, "rand 1:", rand_1, "rand 2:", rand_2)

        v_current = population[i, 2:4]
        p_best = best_locals[i, 0:2]
        x = population[i, 0:2]
        g = best_global[0:2]
        v_next = w*v_current + phi_1 * rand_1 * ( p_best - x ) + phi_2 * rand_2 * ( g - x )
        x_next = x + v_next
        new_fitness = sample_fitness( x_next )

        print("new position:", x_next, "new velocity:", v_next, "new fitness:", new_fitness)

        if new_fitness < population[i, 4]:
            best_locals[i, 0:2] = x_next
            best_locals[i, 2] = new_fitness

        if new_fitness < best_global[2]:
            best_global[0:2] = x_next
            best_global[2] = new_fitness

        population[i, 0:2] = x_next 
        population[i, 2:4] = v_next
        population[i, 4] = new_fitness

         
    population_df = pd.DataFrame(data=population, columns=['x', 'y', 'v_x', 'v_y', 'fitness'])
    print("\nPopulation:")
    print(population_df)

    best_locals_df = pd.DataFrame(data=best_locals, columns=['x', 'y', 'fitness'])
    print("\nbest locals:\n", best_locals_df)
    print("\nbest global:", best_global)


population = population[population[:, vector_size].argsort()]
population_df = pd.DataFrame(data=population, columns=['x', 'y', 'a', 'b', 'fitness'])
print("\nLast population:")
print(population_df)

print("..................................................")
print("\nRESULT..........................................")
print("Best fitness:",  best_global[2])
print("Solution:",  best_global[0:2])