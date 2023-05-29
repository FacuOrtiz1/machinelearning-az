# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:14:43 2023

@author: Faaor
"""

# Upper Confidence Bound (UCB)

# Importanción de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Cargamos el dataset 
# dataset= pd.read_csv("C:/Users/Faaor/OneDrive/Escritorio/Curso ML/machinelearning-az/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")
dataset= pd.read_csv("D:/FACU/machinelearning-az/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")
# El dataset se compone de anuncios de publicidad 10 anuncios de publicidad y lo que queremos ver es cual es el mejor de todos
# En este caso, al igual que con clustering, no tenemos un proceso de clasificación ni sabemos cual va a ser el mejor caso. Son todas variables independientes. 
# El dataset representa si el usuario hizo click o no y en base a esta información vamos a construir los intervalos de confianza. 
# Vamos a mostrar los anuncios de forma aleatoria para los distintos usuarios y vamos a observar su respuesta. Si el usuario hace click en el anuncio le damos una recompensa igual a 1 y si no hace la recompensa es 0.
# Esto para 10mil usuarios de la red. En base a esto construimos el intervalor de confianza para cada uno de los anuncios. 
# El problema de hacerlo al azar es que un anuncio puede ser más mostrado que otro, entonces habrá una estrategicamente diseñada para que lo que se muestre dependa de los resultados obtenidos previamente. 
# Las rondas en las que se va a mostrar los anuncios a los usuarios son 10mil. 
# Las recompenass obtenidas se van a sumar con el que se va a fabricar el intervalo de confianza.
# Si los anuncios se mostraran de forma aleatoria a los usuarios no se obtienen tan buenos resultados. Por este motivo se usas un algoritmo que seleccione de manera no aleatoria esto. 

# # Implementrar una Selección Aleatoria
# import random
# N = 10000
# d = 10
# ads_selected = []
# total_reward = 0
# for n in range(0, N):
#     ad = random.randrange(d)
#     ads_selected.append(ad)
#     reward = dataset.values[n, ad]
#     total_reward = total_reward + reward

# # Visualizar los resultados - Histograma
# plt.hist(ads_selected)
# plt.title('Histograma de selección de anuncios')
# plt.xlabel('Anuncio')
# plt.ylabel('Número de veces que ha sido visualizado')
# plt.show()
dimensionDataset = np.shape(dataset)
d = 10 # Número de anuncios 
numberOfSelections = [0] * d # De esta manera le doy la dimensión a mi vector de ceros. Cantida de veces que cada anuncio fué seleccionado 
sumsOfRewards = [0] * d # Suma de recompensas. Obtiene la cantidad de recompensas que hay recibido en cada anuncio. 
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
    maxUpperBound = 0 # Inicializamos con cada ronda lo que sería el máximo intervalo de confianza superior. Así puedo almacenar cual fue el mejor anuncio durante toda la ronda. 
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