# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:30:12 2023

@author: Faaor
"""

# Muestreo Thompson

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Cargamos el dataset 
# dataset= pd.read_csv("C:/Users/Faaor/OneDrive/Escritorio/Curso ML/machinelearning-az/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")
dataset= pd.read_csv("D:/FACU/machinelearning-az/datasets/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling/Ads_CTR_Optimisation.csv")

dimensionDataset = np.shape(dataset)
d = 10 # Número de anuncios 

# Se usa un vector para que en cada vuelta se determine el número de elementos que reciben recompensa y los que no
number_of_rewards_1 = [0] * d # Numero de veces que obtuvimos un 1 en el anuncio por cada ronda
number_of_rewards_0 = [0] * d # Numero de veces que obtuvimos un 0 en el anuncio por cada ronda

var = []

addsSelected = []  # Este array contiene el anuncio seleccionado en cada ronda
totalReward = 0

for i in range(0,dimensionDataset[0]): # Para cada una de las rondas en los que se corren los anuncios 
    max_random = 0 # Para cada ronda tenemos que seleccionar el máximo valor aleatorio posible de la  ronda  
    ad = 0 # condición inicial al comenzar la ronda 
    
    for j in range(0,d): # Con este for se analiza cada uno de los 10 anuncios para cada persona  
        random_beta = random.betavariate(number_of_rewards_1[j]+1,number_of_rewards_0[j]+1) # Formula del paso 2
        var.append(random_beta)
        if random_beta > max_random: # Chequeo si el maximo valor aleatorio que calculé antes es mayor que el máximo valor random almacenado. 
            max_random = random_beta
            ad = j # Almaceno el anuncio con mayor intervalor superior para la ronda.
                    
    addsSelected.append(ad) # Almaceno el anuncio seleccionado 
    reward = dataset.values[i,ad] # Dado ese anuncio óptimo en la ronda, buscamos que tipo de recompensa le dimos, es decir si hizo click o no . 
    
    # Incrementamos el click o no click en uno 
    if reward == 1:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else: 
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
        
    totalReward = totalReward + reward

# Histograma de resultados
plt.figure(1)
plt.hist(addsSelected)
plt.title("Histograma de anuncios")
plt.xlabel("N° de anuncio")
plt.ylabel("Frecuencia de selección de anuncio")
plt.show()


