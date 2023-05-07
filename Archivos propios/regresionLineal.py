#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:11:34 2020

@author: Facundo Ortiz
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
dataset = pd.read_csv("D:/FACU/machinelearning-az/datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
X = dataset.iloc[:, :-1].values  
Y = dataset.iloc[:, -1:].values 

# Interesante: No se puede aplicar la regresión si el conjunto X no es un array. Y no debe necesariametne ser uno vector

# Dividir el data set en conjunto de entrenamiento y en conjunto de testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = 1/3, random_state = 0)

# Uno de los parametros de regression.fit admite realizar la normalización directamente en el método por lo que no sería necesario realizarlo por nosotros.

regression = LinearRegression()
regression.fit(X_train, Y_train) # fit se utiliza para ajustar o crear el modelo de regresión lineal

# Predecir el conjunto de test.
# En la fase de entrenamiento, el modelo aprende las relaciones y luego en la fase
# en la fase de test se prueba que tan bien realiza la estimación y calificar el modelo.
Y_pred = regression.predict(X_test)

# Visualización de los resultados de la regresión lineall
plt.scatter(X_train, Y_train, color = "red") # Agrego la nube de puntos como una dispersión
plt.plot(X_train, regression.predict(X_train), color = "blue") # Agrego la linea de la regresión
plt.title("Sueldos vs Años de experiencia(Conjunto de training)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo")
plt.show()

# Visualización de los resultados de la regresión lineal sobre el conjunto de prueba
plt.scatter(X_test, Y_test, color = "red") # Agrego la nube de puntos como una dispersión
plt.plot(X_train, regression.predict(X_train), color = "blue") # Agrego la linea de la regresión
plt.title("Sueldos vs Años de experiencia(Conjunto de testing)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo")
plt.show()

