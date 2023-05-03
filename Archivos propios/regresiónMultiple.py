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
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
 
# Importar el Data set
dataset = pd.read_csv("C:/Users/Faaor/OneDrive/Escritorio/Curso ML/machinelearning-az/datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv")
# Defino mi matríz  de caracteristicas. X seria la matriz de caracteristicas e Y lo que quiero predecir. 
X = dataset.iloc[:, :-1].values  
Y = dataset.iloc[:, -1:].values 



#No es necesario aplicar LabelEnconder para crear las variables dummy.
transformer = [('one_hot_encoder', OneHotEncoder(categories='auto'), [-1])] # Creo el transformador de mi columna. 
ct = ColumnTransformer(transformer, remainder='passthrough') # Creo instancia de la clase. 
X_t = np.array(ct.fit_transform(X), dtype=np.float)

#Evitar la trampa de las variables ficticias. 
X_t = X_t[:,1:]
# Interesante: No se puede aplicar la regresión si el conjunto X no es un array. Y no debe necesariametne ser uno vector

# Dividir el data set en conjunto de entrenamiento y en conjunto de testing

X_train, X_test, Y_train, Y_test = train_test_split(X_t, Y,test_size = .2, random_state = 0)

# Ajustar el modelo de regresión lineal múltiple con el conjunto de entrenamiento
regretion = LinearRegression()
regretion.fit(X_train, Y_train)

#Predicción de los resultados en el conjunto de testing. Para esto vamos a usar los datos de testing
y_pred = regretion.predict(X_test)