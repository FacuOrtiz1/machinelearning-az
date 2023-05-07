# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 21:52:08 2023

@author: Faaor
"""

# Plantilla de Pre Procesado

# Cómo importar las librerias en Python
from regresiónHaciaAtrasAuto import backwardElimination, backwardElimination_R
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm #Eliminación hacia atras. Esta librería considera que la columna de 1s es el termino independiente. 
 
# Importar el Data set
#dataset = pd.read_csv("C:/Users/Faaor/OneDrive/Escritorio/Curso ML/machinelearning-az/datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv")
dataset = pd.read_csv("D:/FACU/machinelearning-az/datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv")
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

#Construnir el modelo óptimo de RLM utilizando la eliminación hacia atrás 
X_t = np.append(arr = np.ones((50,1)).astype(int), values = X_t , axis = 1) # Axis = 1 lo agrega como columna, 0 es fila.
SL = 0.05 #P-Valor toletado

# Se crea un nuevo objeto regresor para aplicar la eliminación hacia atrás.
# Si bien en LinearRegression creo un regresor con todas la variables del modelo, este se creo 
# con el fin de crear un modelo de regresión múltiple y ajustarlos con datos de entrenamiento y validarlo con
# datos de test. La lib de statsmodels necesita volver a crear ese modelo de nuevo. Es otro objeto
# por lo que es necesario volver a crear el objeto de regresión pero con la nueva estructura que se genero en X_t

X_opt = X_t[:,[0,1,2,3,4,5]]#En esta variable se van a almacenar las variables estadisticamente significativas. Pra comenzar X_opt toma todas las variables.
# regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# #endog = variables endojenas o que se buscan estimar
# #exog = variables exogenas o variables utilizadas para estimar. 

# regression_OLS.summary()

X_opt1 = backwardElimination(X_opt, Y, SL)
X_opt2 = backwardElimination_R(X_opt, Y, SL)