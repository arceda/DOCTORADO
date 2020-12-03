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


def sample_fitness(sample):
    x = sample[0]
    y = sample[1]
    #a = sample[2]
    #b = sample[3]
    #return (x + 2*y -7)**2 + (2*x + y - 5)**2 + (a + 2*b - 7)**2 + (2*a + b - 5)**2
    return (x + 2*y -7)**2 + (2*x + y - 5)**2 

def compute_fitness(population):
    for sample in population:        
        sample[vector_size] = sample_fitness(sample)

def build_population(sequences, k, max_length, min_length, population_size):
    population = []
    for index in range(population_size):    
        align_seqs_length = random.randint(max_length, int(1.5*max_length) )
        #print(align_seqs_length)
        total_gaps = []
        for seq in sequences:
            seq_len = len(seq[1])        
            gaps = []
            for i in range( align_seqs_length - seq_len ):
                gaps.append(random.randint(0, seq_len))
            total_gaps.append( gaps )  

        population.append( total_gaps )

    #population = np.array( population )    
    return population
    #print(total_gaps)

def print_solution(particle, sequences):
    #print(particle, sequences)
    for i in range( len(particle) ):
        #print( particle[i], sequences[i] )
        gaps = particle[i].copy()
        gaps.sort()
        align_seq = split(sequences[i][1])
        #print(gaps, align_seq)
        for gap in gaps:
            align_seq.insert( gap, '-' )

        print(align_seq)
        
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
population_size = 10
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
print("attributes per seq (n) :", att_per_seq)
print("seq number (k) :", k)
print("vector_size (n*k) :", vector_size)
print("rand_1 and rand_2 = [0.1]")
print("phi_1 and phi_2 = 2")
print("Iterations:", iterations)
print("...............................................................\n")


population = build_population(sequences, k, max_length, min_length, population_size)
align = print_solution( population[0], sequences )

sys.exit(0)


positions = np.random.randint(att_per_seq, size=(population_size, vector_size))
positions = np.random.randint(2, size=(population_size, vector_size))

print(positions)
print(positions.shape)

sys.exit(0)

velocities = np.random.uniform(-1, 1, size=(population_size, 3))
population = np.hstack( (positions, velocities) )

compute_fitness(population)

population = population[population[:, vector_size].argsort()]

best_locals = population[ :, 0:2 ].copy()
best_locals = np.hstack( ( best_locals, population[ :, vector_size ].reshape( population_size, 1 ).copy() ) )

best_global = best_locals[0]

population_df = pd.DataFrame(data=population, columns=['x', 'y', 'v_x', 'v_y', 'fitness'])
best_locals_df = pd.DataFrame(data=best_locals, columns=['x', 'y', 'fitness'])
print("population:\n", population_df)
print("\nbest locals:\n", best_locals_df)
print("\nbest global:", best_global)

for iter in range(iterations):
    print("\n***  ITERATION ", iter, "**************************************************************************")
    print("********************************************************************************************\n")

    for i in range( population_size ):
        print( "\nParticle ", i, "=", population[i], "***********************************" )
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