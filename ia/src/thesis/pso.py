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

import warnings
warnings.filterwarnings("ignore")

from numpy.random import seed
pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', 1000)
np.set_printoptions(linewidth=200)

#seed(1)
#random.seed(1)

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
        align_seqs_length = random.randint(max_length, int(1.8*max_length) )
        # tods con el mismo tamanio
        #align_seqs_length = int(1.2*max_length) 

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
        score = MSA_score(total_seqs_align)

        # compute fitness
        '''
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
        '''
        particle[0] = score
    
#multiple sequence alignment score
def MSA_score(sequences):
    
    total_score = 0
    num_seqs = len(sequences)
    for i in range(num_seqs):
        for j in range( i+1, num_seqs ):
            #print( "i, j", i, "-", j )
            total_score += pairwise_alignment_score(sequences[i], sequences[j])

    #print("MSA SCORE")
    #print(sequences)   
    #print(total_score)

    return total_score

def pairwise_alignment_score(seq1, seq2):
    opening_gap = 1
    gap_extension = 1
    equality = 2

    score = 0
    for i in range(len(seq1)):
        if seq1[i] == seq2[i] and seq1[i] != '-':
            score += equality
        elif seq1[i] == seq2[i] and seq1[i] == '-':
            score += 0
        elif seq1[i] == '-' or seq2[i] == '-':
            score -= gap_extension

    return score

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

def matrix_to_particle(particle): # redibe una matrizx y devuelve una lista de string [ 'ACTGT' ATCC' ... ]
    seqs = []
    for sol in particle:
        seq_str = ''.join(sol)
        seqs.append( seq_str )
    return np.array(seqs).astype(object)

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

    #print("removing bases ( n =",n,") original:", seq, "result:", result, "indices:", indices)    
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
    #print("adding bases ( n =",n,") original:", right, "result:", result, "indices:", indices, "values:", values)    
    return result

def check_size(particle):
    #print("particle before padding gap:\n", particle, particle.shape)
    # insertamos padding gaps al final para que todas las seq tengan el mismo tamanio
    lengths = []
    for sol in particle:
        lengths.append( sol.shape[0] )
        #lengths.append( len(sol) )

    max_len = np.argmax( lengths )
    
    for i in range(particle.shape[0]):
        if particle[i].shape[0] < lengths[max_len]:
            for j in range( lengths[max_len] - particle[i].shape[0] ):
                particle[i] = np.insert( particle[i], particle[i].shape[0], '-' )
                #print("after insert gap:", particle[i])

    # verificamos si en la ultima comlumna todos son gaps, si es asi, eliminamos    
    particle = np.stack( particle, axis=0 )
    #print("particle after padding gap:\n", particle, particle.shape)
    #print("particle:", particle, particle.shape)
    while True:
        last_col = particle[ :, particle.shape[1] - 1 ]
        #print("last_col:", last_col)
        if np.all(last_col == '-'):
            particle = particle[:, 0:particle.shape[1] - 1]
        else:
            break


    #print("particle after padding gap 2:\n", particle, particle.shape)
    return particle

# ejemplo crosovers
"""
['W' 'G' 'K' 'V' '-'     '-' 'N' 'V' 'D']  	    W1=3
['W' 'G' 'K' '-' '-'     'V' 'N' 'V' 'D']   	W2=4
['W' 'G' 'K' 'V' '-'     'N' 'V' 'D']


['W' 'G' 'K' '-' '-'     'V' 'N' 'V' 'D']	    W1=4
['W' 'G' 'K' 'V' '-'     '-' 'N' 'V' 'D']   	W2=3
['W' 'G' 'K' '-' '-'     'V' '-' 'N' 'V' 'D']
"""
def crossover_v2(particle_1, particle_2): # particle_1=leader, particle_2 = particle
    print("\nCROSSOVER: \n", particle_1, "=> leader", "\n", particle_2, "=> particle")
    
    dist = distance(particle_1, particle_2)
    if dist == 0: # si es la misma secuencia
        return particle_1
    
    
    #print(int(len(particle_1[0])/2), int(dist*max(len(particle_1[0]), len(particle_2[0])) - 2))
    #r_min = int(len(particle_1[0])/3)
    r_min = 1
    r_max = int(dist*max(len(particle_1[0]), len(particle_2[0])) - 1)
    cross_point = random.randint( min(r_min, r_max), max(r_min, r_max))
    

    print(" dist = ", dist, "crosspoint =", cross_point)

    if dist > 0.5:
        particle_1 = particle_to_matrix(particle_1)
        particle_2 = particle_to_matrix(particle_2)
    else:
        particle_1 = particle_to_matrix(particle_2)
        particle_2 = particle_to_matrix(particle_1)
    
    child = []

    for i in range(particle_1.shape[0]): 
        #seq_1_gaps, seq_2_gaps = get_gap_indices(particle_1, particle_2)
        seq_1_gaps = np.where(particle_1[i] == '-')[0]
        seq_2_gaps = np.where(particle_2[i] == '-')[0]
        child_gaps = []
        child_seq_without_gaps = np.delete(particle_1[i], seq_1_gaps)
        
        if seq_1_gaps.shape[0] > 0:                        
            print("\npar_1 => ",particle_1[i], "---- gaps:", seq_1_gaps, "\npar_2 => ", particle_2[i], "---- gaps:",  seq_2_gaps)
            child_gaps = seq_1_gaps[seq_1_gaps <= cross_point]
            for gap in seq_2_gaps:
                if gap > cross_point:
                    child_gaps = np.insert(child_gaps,child_gaps.shape[0], gap)
            
            # insertamos los hgaps en el hijo
            
            child_seq = child_seq_without_gaps.copy()
            for gap in child_gaps:
                if gap > child_seq.shape[0]:
                    child_seq = np.insert(child_seq, child_seq.shape[0], '-')    
                else:
                    child_seq = np.insert(child_seq, gap, '-')
                
            print("child => ", child_seq, "---- gaps:", child_gaps)
            
            child.append(''.join(child_seq))

        else:
            
            child.append(''.join(particle_1[i]))

    # falta agregar que todas las seq tengan el mismo tamanio
    max_length = len(max(child, key=len))
    for i in range(len(child)):
        length = len(child[i])
        for j in range( max_length - length ):
            child[i] = child[i] + '-' 

    return np.array(child).astype(object)

"""
['W' 'G' 'K' 'V' '-'     '-' 'N' 'V' 'D']  	    W1=3
['W' 'G' 'K' '-' '-'     'V' 'N' 'V' 'D']   	W2=4
['W' 'G' 'K' 'V' '-'     'N' 'V' 'D']


['W' 'G' 'K' '-' '-'     'V' 'N' 'V' 'D']	    W1=4
['W' 'G' 'K' 'V' '-'     '-' 'N' 'V' 'D']   	W2=3
['W' 'G' 'K' '-' '-'     'V' '-' 'N' 'V' 'D']
"""
def crossover(particle_1, particle_2): # particle_1=leader, particle_2 = particle
    print(particle_1, "\n", particle_2)
    dist = distance(particle_1, particle_2)*len(particle_1[0])
    
    seq_size_1 = len(particle_1[0])
    seq_size_2 = len(particle_2[0])

    cross_point = random.randint(0, min(seq_size_1, seq_size_2) - 2)

    if dist > 0.5: # the leader will provide the largest segment
        if cross_point >= int(len(particle_1[0])/2): #el segmento mas largo es particle 1
            particle_1 = particle_to_matrix(particle_1)
            particle_2 = particle_to_matrix(particle_2)  
        else: #el segmento mas largo es particle 2
            particle_2 = particle_to_matrix(particle_1)
            particle_1 = particle_to_matrix(particle_2) 
    else: # the leader will provide the shortest segment
        if cross_point >= int(len(particle_1[0])/2): #el segmento mas largo es particle 1
            particle_2 = particle_to_matrix(particle_1)
            particle_1 = particle_to_matrix(particle_2)  
        else: #el segmento mas largo es particle 2
            particle_1 = particle_to_matrix(particle_1)
            particle_2 = particle_to_matrix(particle_2) 


    #particle_1 = particle_to_matrix(particle_1)
    #particle_2 = particle_to_matrix(particle_2)     

    #print(particle_1, '\n\n', particle_2, '\n')
    #particle_2[:, 0:cross_point] = particle_1[:, 0:cross_point]
    #print(particle_1, '\n\n', particle_2, '\n')

    #print("particle_2:\n", particle_2, particle_2.shape)

    new_particle_2 = []
    for i in range(particle_1.shape[0]): 
        print("crossover between ( point = ", cross_point, ")\n", particle_1[i], '\n', particle_2[i])

        # algoritmo antiguo, tiene errores
        lp_1 = particle_1[i][0:cross_point]
        rp_1 = particle_1[i][cross_point:particle_1[i].shape[0]] 

        lp_2 = particle_2[i][0:cross_point]
        rp_2 = particle_2[i][cross_point:particle_2[i].shape[0]] 

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
        
        print("new_seq", new_seq, new_seq.shape, '\n')
        new_particle_2.append( new_seq )
        
    
    new_particle_2 = check_size( np.array(new_particle_2) )
    print('fin crossover\n***********************************\n')
    return matrix_to_particle(new_particle_2)


# this mutation insert a gap in a ramdon position in a random particle
def mutate_particle(particle):
    num_particles = particle.shape[0]
    pos = random.randint(1, num_particles - 1)   
    gap_position = random.randint(0, len(particle[pos]) - 1)

    #despues de inseretar un gap, debo insertar otro en los demas para que tengan la misma longitud
    #0 insertamos al principio, 1=insertamos al final
    fill_position = random.randint(0, 1) 

    print( "\nMUTATION particle pos:", pos, " gap at: ", gap_position, "************************************")
    print("particle\t\t", particle)

    for _i in range(1, num_particles):        
        if _i == pos:
            print(particle[pos][:gap_position], particle[pos][gap_position:])
            particle[pos] = particle[pos][:gap_position] + "-" + particle[pos][gap_position:]
        else:
            if fill_position == 0:
                particle[_i] = "-" + particle[_i]
            else:
                particle[_i] =  particle[_i] + "-" 
    
    print("particle mutated\t", particle)


# revisamos si hay gaps al inicio y al final que se puedan eliminar
def clean_particles(particle):
    seqs = particle[1:particle.shape[0]]       

    # cast string to list pra tratarlo como una matriz
    seq_vec = []
    for seq in seqs:
        seq_list = split(seq)
        seq_vec.append( np.array(seq_list) )

    seq_vec = np.array(seq_vec)
    #print( seq_vec , seq_vec.shape)
    rows = seq_vec.shape[0]
    cols = seq_vec[0].shape[0]
    seq_vec = np.concatenate( seq_vec, axis=0 )
    seq_vec = np.reshape( seq_vec, (rows, cols) )

    #cols = total_seqs_align.T
    print("\nCLEANING ")
    print( particle )
    print( seq_vec , seq_vec.shape)

    while True:
        col = seq_vec[ :, 0 ]
        if np.all(col == '-'):
            print("remove first col")
            seq_vec = seq_vec[ :, 1:seq_vec.shape[1] ]
        else:
            break

    while True:
        col = seq_vec[ :, seq_vec.shape[1]-1 ]
        if np.all(col == '-'):
            print("remove last col")
            seq_vec = seq_vec[ :, 0:seq_vec.shape[1] - 1 ]
        else:
            break

    particle_ = matrix_to_particle(seq_vec)
    particle_ = np.insert(particle_, 0, None)

    print( particle_ )
    return particle_

sequences, k, max_length, min_length = read_sequences(current_dir + "/seqs/S/")
#sequences, k, max_length, min_length = read_sequences_s8()
max_gaps_allowed = math.floor(0.3*max_length)
gaps_allowed = np.random.randint(max_gaps_allowed+1, size=sequences.shape[0])

print(sequences)
#print(sequences.shape)
print(k, max_length, min_length)
#print(max_gaps_allowed, gaps_allowed)

###########################################################################################################
###########################################################################################################3333
# Parametros
population_size = 30
att_per_seq = int(max_length*0.2)   # gaps dentro de una particula para cada secuencia
vector_size = k*att_per_seq         # 0.2 porque solo se tiene como maximo un 20% de gaps
mutation_rate = 0.2
iterations = 20
simulations = 30

num_test = int(sys.argv[1])

###########################################################################################################
###########################################################################################################
print("\n...............................................................")
print("PARAMETERS:")
print("population_size:", population_size)
print("Iterations:", iterations)
print("...............................................................\n")



#sys.exit(0)

best_fitnes = []
for test in range(num_test):

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
    print("\nbest_global:\n", best_global)

    #dist = distance(np.array(['WGKV--NVD', 'WDKV--N--', 'S-KVGGN--']), np.array(['WGK--VNVD', '-WDK---VN', 'S-KVGG-N-']))
    #print(dist)

    #crossover(np.array(['WGKV--NVD', 'WDKV--N--', 'S-KVGGN--']), np.array(['WGK--VNVD', '-WDK---VN', 'S-KVGG-N-']))
    #crossover(np.array(['WGK--VNVD', 'WDKV--N--', 'S-KVGGN--']), np.array(['WGKV--NVD', '-WDK---VN', 'S-KVGG-N-']))


    for iter in range(iterations):
        print("\n***  ITERATION ", iter, "**************************************************************************")
        print("********************************************************************************************\n")

        new_population = []
        for i in range( population_size ):
            print( "\nParticle ", i, "***********************************" )
            
            new_particle = crossover_v2( best_global[1:population.shape[1]], population[i, 1:population.shape[1]] )
            #new_particle = crossover( best_global[1:population.shape[1]], population[i, 1:population.shape[1]] )
            new_particle = np.insert(new_particle,0, None)
            

            print("\nnew particle: ", new_particle)
            #sys.exit(0)
            r = random.uniform(0,1)
            if r > mutation_rate:
                mutate_particle(new_particle)

            new_particle = clean_particles(new_particle)

            new_population.append(new_particle)
        

        new_population = np.array(new_population).astype(object)

        population = new_population
        evaluate_solution(population)
        population_df = pd.DataFrame(data=population)
        print("\nNew Population:\n", population_df)

        best_global_index = np.argmax(population[:,0])
        best_global = population[best_global_index]
        print("\nBest global:\n", best_global)
        print( "Score:", best_global[0] )
        
        for z in range( 1, best_global.shape[0] ):
            print(">", z)
            print(best_global[z])

    best_fitnes.append( best_global[0] )



print("\nsolutions: ", best_fitnes, " mean: ", np.mean( np.array(best_fitnes) ))
print("\n", sequences)
# here it is a MSA viewer
# https://www.ebi.ac.uk/Tools/msa/mview/


    


