# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:43:26 2023

@author: Equipo
"""

# A priori

#Importo las librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importo el dataset
dataset = pd.read_csv("C:/Users/Faaor/OneDrive/Escritorio/Curso ML/machinelearning-az/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori/Apriori_Python/Market_Basket_Optimisation.csv",header=None)
# Con este analisis se busca comprender las reglas de asociación de los producto para comprender como distribuir correctamente mis productos dentro de la tienda y que de esta manera los clientes compren más

dimentions = np.shape(dataset)
transactions = []
for i in range(0,dimentions[0]):
    transactions.append([str(dataset.values[i,j]) for j in range(0,dimentions[1])])


# Entrenar el algoritmo APriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence= 0.2, min_lift = 3, min_length= 2)
# Se necesita especificar el soporte mínimo y el nivel de confianza mínimo.