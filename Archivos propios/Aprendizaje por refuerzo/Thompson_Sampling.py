# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:30:12 2023

@author: Faaor
"""

# Muestreo Thompson

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Cargamos el dataset 
# dataset= pd.read_csv("C:/Users/Faaor/OneDrive/Escritorio/Curso ML/machinelearning-az/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")
dataset= pd.read_csv("C:/Users/Faaor/OneDrive/Escritorio/Curso ML/machinelearning-az/datasets/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling/Ads_CTR_Optimisation.csv")

dimensionDataset = np.shape(dataset)
d = 10 # Número de anuncios 
number_of_rewards_1 = [0] * d # Numero de veces que obtuvimos un 1 en el anuncio
number_of_rewards_0 = [0] * d # Numero de veces que obtuvimos un 0 en el anuncio
addsSelected = []  # Este array contiene el anuncio seleccionado en cada ronda
totalReward = 0

''' Durante las 10 primeras rondas, sumsOfRewards y numberOfSelection son cero. Los anuncios están empezando a ser mostrados y tengo que trabajar con condiciones iniciales que
son nulas. Hay que acotar el algoritmo, para que las primeras rondas solo se recabe información y luego de haber recabado la información básica inicial,
se pueda comenzar a aplicar este algoritmo. Para poder actuar, necesitamos que el número de selecciones no sea cero. 


En las primeras 10 pasadas del bucle exterior, se pasara por cada anuncio estableciendo la cota superior del intervalo de confianza en 
1e400. De esta manera se establece en el mismo valor a todos las cotas. Esto es necesario para poder comenzar a ejecutar las ecuaciones
del modelo. 
'''
for i in range(0,dimensionDataset[0]): # Para cada una de las rondas en los que se corren los anuncios 
    max_random = 0 # Inicializamos con cada ronda lo que sería el máximo intervalo de confianza superior. Así puedo almacenar cual fue el mejor anuncio durante toda la ronda. 
    ad = 0 # Tenemos la condición inicial.
    for j in range(0,d): # Con este for se analiza cada uno de los 10 anuncios para cada persona  
        if (numberOfSelections[j]>0): # Las primeras 10 selecciones de los usuarios son descartadas. 
            averageReward = sumsOfRewards[j] / numberOfSelections[j] # Raya roja punteada
            deltaI = math.sqrt(3/2 * math.log(i+1)/numberOfSelections[j]) 
            upperBound = deltaI + averageReward # Intervalo de confianza superior 
        else:
            upperBound = 1e400 # Establezco la confianza máxima para los primeros en un valor muy alto. Esto no influye en el algoritmo. 
            # Establecemos un upperBound tan grande porque qeuremos que cuando un anuncio entra y le calculamos su upper bound, hasta que todo hata sido inicializado, me interesa que todos los anuncios hayan sido seleccionados como el de frontera superior. 
        if upperBound > maxUpperBound: # Chequeo si el intervalo que calculé antes es mayor que el máximo intervalor superior almacenado. 
            maxUpperBound = upperBound
            ad = j # Almaceno el anuncio con mayor intervalor superior para la ronda. 
    addsSelected.append(ad) # Almaceno el anuncio seleccionado 
    numberOfSelections[ad] = numberOfSelections[ad] + 1 # Actualizamos el número selecciones
    reward = dataset.values[i,ad] # Dado ese anuncio óptimo en la ronda, buscamos que tipo de recompensa le dimos, es decir si hizo click o no . 
    sumsOfRewards[ad] = sumsOfRewards[ad] + reward # Actualizamos la suma de recompensas para cada anuncio particular. Las recompensas no dependen del usuario sino que se analizan para todos, solo indexamos por el anuncio más importante para la ronda. 
    totalReward = totalReward + reward

# Histograma de resultados
plt.hist(addsSelected)
plt.title("Histograma de anuncios")
plt.xlabel("N° de anuncio")
plt.ylabel("Frecuencia de selección de anuncio")
plt.show()