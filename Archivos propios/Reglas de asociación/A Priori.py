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
# dataset = pd.read_csv("C:/Users/Faaor/OneDrive/Escritorio/Curso ML/machinelearning-az/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori/Apriori_Python/Market_Basket_Optimisation.csv",header=None)
dataset = pd.read_csv("D:/FACU/machinelearning-az/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori/Apriori_Python/Market_Basket_Optimisation.csv",header=None)
# Con este analisis se busca comprender las reglas de asociación de los producto para comprender como distribuir correctamente mis productos dentro de la tienda y que de esta manera los clientes compren más

dimentions = np.shape(dataset)
transactions = []
for i in range(0,dimentions[0]):
    transactions.append([str(dataset.values[i,j]) for j in range(0,dimentions[1])])
# La anterior es una lista de listas, donde cada lista contiene la fila completa de elementos. 

# Entrenar el algoritmo Apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence= 0.2, min_lift = 3, min_length= 2)
# Se necesita especificar el soporte mínimo y el nivel de confianza mínimo. 3(comprados por día) * 7 dias / 7500 (Número de compras totales)
# min support = en que porcentaje de la cesta de compras aparece el item para que se tenga en cuenta. Con que presencia mínima debe aparecer un item para ser consiuderado como el objetivo en la regla de asociación. Me interesan items que se compran 3 o 4 veces al día y no esos que se compran una vez a la semana. 
# min confidence = en que porcentaje de la cesta de compras tiene que aparecer los items en conjunto para tenerse en cuenta. 
# min lift = Considera las reglas más importantes. 
# min length = minimo de productos juntos en la cesta de compra que deriven en la compra de uno al otro

# Visualización de las reglas
results = list(rules)
# Las reglas que tengan lift más elevado van a ser la más relevantes y que permiten clasificar mejor los items. 
for i in range(0,10):
    print(results[i])