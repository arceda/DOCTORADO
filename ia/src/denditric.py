import numpy as np
import random
import scipy.stats
import pandas as pd
import sys
np.set_printoptions(precision=3, suppress=True)
import math
from sklearn.metrics import accuracy_score

#random.seed(1)
from numpy.random import seed
#seed(1)
pd.set_option('display.max_rows', None)

###########################################################################################################
###########################################################################################################3333
# Parametros
at = 0.3
migrations = 50
iterations = 10000

###########################################################################################################
###########################################################################################################

print("parameters.....................................................")
print("Atributos: CS, CH, BH, NN")
print("Migrations:", migrations)
print("Iterations:", iterations)
print("at:", at)
print("...............................................................\n")

dataset = pd.read_csv('denditric_breast-cancer-wisconsin.data') 
dataset.columns = ['id', 'CT', 'CS', 'CH', 'MA', 'SE', 'BH', 'BC', 'NN', 'MI', 'Class']

data = []
for row in dataset.values:
    if row[1] != '?' and row[2] != '?' and row[3] != '?' and row[6] != '?' and row[8] != '?':
        data.append( [row[1], row[2], row[3], row[6], row[8], row[10] ] )
    
data = np.array(data).astype(float)

data_pd = pd.DataFrame(data=data, columns=["CT", "CS", "CH", "BH", "NN", "CLASS"])
print("Data:\n", data_pd)

weights = np.array(  [[2, 0, 2], [1, 0, 1], [0.5, 1.2, 0.5]] )
weights_pd = pd.DataFrame(data=weights, columns=["PAMP", "SS", "DS"], index=["CSM", "smDC", "mDC"])
print("\nweights:\n", weights_pd)


mediam_ct = np.median( data[:, 0] )
mediam_cs = np.median( data[:, 1] )
mediam_ch = np.median( data[:, 2] )
mediam_bh = np.median( data[:, 3] )
mediam_nn = np.median( data[:, 4] )

mean_ct = np.mean( data[:, 0] )
mean_cs = np.mean( data[:, 1] )
mean_ch = np.mean( data[:, 2] )
mean_bh = np.mean( data[:, 3] )
mean_nn = np.mean( data[:, 4] )

print("\nStatistics:")
print("CT mediam:", mediam_ct, " mean: ", mean_ct)
print("CS mediam:", mediam_cs, " mean: ", mean_cs)
print("CH mediam:", mediam_ch, " mean: ", mean_ch)
print("BH mediam:", mediam_bh, " mean: ", mean_bh)
print("NN mediam:", mediam_nn, " mean: ", mean_nn)

###################################################################################
# dataset signal 
###################################################################################

signal_data = np.zeros((data.shape[0], 7 ))  # las 4 col adicionales es para guardar cuantas veces aparece, cuantes veces es maduro, el MCAV y la clase

for i in range(data.shape[0]):
    if data[i,0] > mediam_ct:
        signal_data[i, 0] = 0.0
        signal_data[i, 1] = abs( data[i, 0] - mean_ct )
    else:
        signal_data[i, 0] = abs( data[i, 0] - mean_ct )
        signal_data[i, 1] = 0.0

signal_data[:, 2] = (np.absolute(data[:, 1] - mean_cs) + np.absolute(data[:, 2] - mean_ch) + np.absolute(data[:, 3] - mean_bh) + np.absolute(data[:, 4] - mean_nn))/4

signal_data_pd = pd.DataFrame(data=signal_data[:, 0:3], columns=["PAMP", "Safe signal", "Danger signal"])
print("\nSignal data set:\n", signal_data_pd)


for iter in range(iterations):
    print("\n***  ITERATION ", iter, "**************************************************************************")
    print("********************************************************************************************\n")

    acc_CSM = 0
    acc_smDC = 0
    acc_mDC = 0

    cache = []
    while acc_CSM < migrations:
        # get ramdom sample
        sample_index = random.randint(0, signal_data.shape[0]-1)
        cache.append(sample_index)     

        acc_CSM     += np.sum(signal_data[sample_index, 0:3]*weights[0])
        acc_smDC    += np.sum(signal_data[sample_index, 0:3]*weights[1])
        acc_mDC     += np.sum(signal_data[sample_index, 0:3]*weights[2])
        
    print( "acc_CSM: ", acc_CSM, " acc_smDC: ", acc_smDC, " acc_mDC: ", acc_mDC )
    print("samples: ", cache)

    
    if acc_smDC > acc_mDC:
        print("class: ", 2.0) #semi madure
        class_tmp = 2.0   
    else:
        print("class: ", 4.0) #madure
        class_tmp = 4.0

    for sample in cache:
        signal_data[sample, 3] += 1 # cuantas veces aparece
        if class_tmp == 4.0: #madure
            signal_data[sample, 4] += 1

# calculamos el MCAV
signal_data[:, 5] = signal_data[:, 4]/signal_data[:, 3] #MCAV

#determinamos la clase en base al MCAV y si supera el threshold
for row in signal_data:
    if row[5] < at:
        row[6] = 2.0
    else:        
        row[6] = 4.0

signal_data_pd = pd.DataFrame(data=signal_data, columns=["PAMP", "SafeSig", "DangerSig", "Nb-antig", "Nb-mat", "MCAV", "classPred"])
print("\nSignal data set:\n", signal_data_pd)


print("\n\nACCURACY: ", accuracy_score(data[:, 5], signal_data[:, 6] ))