# -*- coding: utf-8 -*-
"""
Created on Mon May  8 21:14:05 2023

@author: Faaor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Importamos el dataset
dataset = pd.read_csv('C:/Users/Faaor/OneDrive/Escritorio/Curso ML/machinelearning-az/datasets/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv')
X = dataset.iloc[:,1].values.reshape(10,1)
Y = dataset.iloc[:,-1].values

"""
# División del dataset en conjunto de test y train 
X_train, Y_train, X_test, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)
"""
"""
# Escalado de variable
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y.reshape(-1,1))
# X_test = sc_X.transform(X_test)
"""

# Ajustamos la regresión con el dataset
regression = DecisionTreeRegressor(random_state=0) # Ver todos los parámetros que usa la formula
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
regression.fit(X, Y)

# Predicción de la regresión 
Y_pred = regression.predict([[6.5]])

#Creamos un nuevo conjunto de puntos para que no quede tan cuadrada la curva
x_grid = np.arange(min(X), max(X), 0.1) # Crea secuencia de datos usando valor min, max y step
x_grid = x_grid.reshape(len(x_grid), 1) #reshape traspone

# Visualización de los resultados con SVR
plt.scatter(X, Y, color = "red") # Agrego la nube de puntos como una dispersión
plt.plot(X, regression.predict(X), color = "blue") # Agrego la linea de la regresión
plt.title("Regresión polinómica")
plt.xlabel("Puesto")
plt.ylabel("Sueldo")
plt.show()