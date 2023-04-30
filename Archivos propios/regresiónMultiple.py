# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 21:52:08 2023

@author: Faaor
"""

# Plantilla de Pre Procesado

# Cómo importar las librerias en Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
 
# Importar el Data set
dataset = pd.read_csv("C:/Users/Faaor/OneDrive/Escritorio/Curso ML/machinelearning-az/datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv")
X = dataset.iloc[:, :-1].values  
Y = dataset.iloc[:, -1:].values 

# Interesante: No se puede aplicar la regresión si el conjunto X no es un array. Y no debe necesariametne ser uno vector

# Dividir el data set en conjunto de entrenamiento y en conjunto de testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 1/3, random_state = 0)