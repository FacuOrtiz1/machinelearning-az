# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:15:56 2023

@author: Equipo
"""

# Bosques aleatorios. 

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Bosques aleatorios. 
dataset = pd.read_csv('D:/FACU/machinelearning-az/datasets/Part 2 - Regression/Section 9 - Random Forest Regression/Position_Salaries.csv')
X = dataset.iloc[:,1].values.reshape((10,1))
Y = dataset.iloc[:,-1].values


# Ajustamos la regresión del modelo
regression = RandomForestRegressor(n_estimators=500, random_state=0)
regression.fit(X,Y) 

# Predicción de nuestro modelo de bosque aleatorio.
y_pred = regression.predict([[6.5]])  


# Creamos un nuevo conjunto de puntos para que no quede tan cuadrada la curva
x_grid = np.arange(min(X), max(X), 0.1) # Crea secuencia de datos usando valor min, max y step
x_grid = x_grid.reshape(len(x_grid), 1) #reshape traspone

# Visualización de los resultados con bosque aleatorio.
plt.scatter(X, Y, color = "red") # Agrego la nube de puntos como una dispersión
plt.plot(x_grid, regression.predict(x_grid), color = "blue") # Agrego la linea de la regresión
plt.title("Regresión polinómica")
plt.xlabel("Puesto")
plt.ylabel("Sueldo")
plt.show()

