# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:21:02 2023

@author: Equipo
"""

# Regresión polinómica

#Cargo librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as train
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # Librería para regresión polinomial

# Los campos position y level son equivalentes, por lo que vamos a trabajar solo con level y salary.

#Importamos el dataset
dataset = pd.read_csv('D:/FACU/machinelearning-az/datasets/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

#En esta receta no se hace la división entre entrenamietno y testing particularmente porque no hay suficientes datos y además solo se tiene un dato por nivel. 

# # Dividimos la data en conjunto de entrenamiento y conjunto de testing
# X_train, Y_train, X_test, Y_test =  train(X, Y, test_size=0.2, random_state=0)

# Ajustar la regresión lineal con el dataset
lineal_regression = LinearRegression() #creamos regresor
lineal_regression.fit(X, Y) #Ajustamos con los datos

# Visualización de los resultados de la regresión lineall
plt.scatter(X, Y, color = "red") # Agrego la nube de puntos como una dispersión
plt.plot(X, lineal_regression.predict(X), color = "blue") # Agrego la linea de la regresión
plt.title("Regresión lineal")
plt.xlabel("Puesto")
plt.ylabel("Sueldo")
plt.show()

# Ajusto regresión polinómica con dataset
polynomial_regression = PolynomialFeatures(degree= 3)
X_poly = polynomial_regression.fit_transform(X) #fit solo crea el modelo y fit transform aplica los cambios al propio objeto.
# Basicamente con la linea anterior se genera la matriz de coeficientes polinomial, usando una columna para las variables independientes
# y el resto de las columnas son las potencias de la varaible independient que seleccionamos. 

# Creamos el objeto de regresión polinomial.
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)
# Basicamente genero el objeto regresión y obtengo los coeficientes para el conjunto

# Creamos un nuevo conjunto de puntos para que no quede tan cuadrada la curva
x_grid = np.arange(min(X), max(X), 0.1) # Crea secuencia de datos usando valor min, max y step
x_grid = x_grid.reshape(len(x_grid), 1) #reshape traspone

# Visualización de los resultados de la regresión lineal
# No se esta cambiando como se ajusto la curva, solo se agregan más puntos para visualizar mejor
plt.scatter(X, Y, color = "red") # Agrego la nube de puntos como una dispersión
plt.plot(x_grid, lin_reg_2.predict(polynomial_regression.fit_transform(x_grid)), color = "blue") # Agrego la linea de la regresión
plt.title("Regresión polinómica")
plt.xlabel("Puesto")
plt.ylabel("Sueldo")
plt.show()

# Predicción para modelo lineal
print(lineal_regression.predict([[6.5]]))

# Predicción para modelo polinómico
print(lin_reg_2.predict(polynomial_regression.fit_transform([[6.5]])))