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

# Cargamos el dataset 
dataset= pd.read_csv("C:/Users/Faaor/OneDrive/Escritorio/Curso ML/machinelearning-az/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")
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

 