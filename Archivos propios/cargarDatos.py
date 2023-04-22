# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:47:37 2023

@author: Equipo
"""

# Plantilla de preprocesado. 

# Importación de librerías

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer# Libreria muy buena para limpieza de datos
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Para dividir los datos
from sklearn.model_selection import train_test_split # La función train_test_split es la función que nos va a permitir dividir el dataset. 
 
# Importar el dataset
dataset = pd.read_csv('D:/FACU/machinelearning-az/datasets/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data.csv')

X = dataset.iloc[:,:-1].values  # Obtengo los valores de dataset de todas las columnas a excepción de la última.
Y = dataset.iloc[:,-1].values

# Tratamiento de los NAs
imputer = SimpleImputer(missing_values = pd.NA, strategy = "mean")
imputer.fit(X[:,1:3]) # Le indicamos que valores con los que va a ajustar
X[:,1:3] = imputer.transform(X[:,1:3]) # Se genera la transformación. Se podría usar fit en otro conjunto de datos y aplicarco sobre X

# Datos categóricos:
# El problema se encuentra en que si yo meto esto a un modelo de ML, este peude interpretar que 0 es mayor que 1 y 2 mayor que 2.
# Por esta razón, como no hay relación u orden entre la numeración asignada y la información del campo, no se debe usar. 
le_X = preprocessing.LabelEncoder() # Creo el codificador de datos. 
X[:,0] = le_X.fit_transform(X[:,0])

le_Y = preprocessing.LabelEncoder() # Creo el codificador de datos. 
y = le_Y.fit_transform(Y)

# Variables ordinales: Se codifican variables que son categóricas pero que tiene un orden o cardinalidad.
#Variables Dummy: Es uan forma de traducir una variable que no tiene un orden o cardinalidad a un conjunto de tantas columas como categorías existen. 
# One hot encoder: Codificación con un solo 1 por fila. 
transformer = [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])] # Creo el transformador de mi columna. 
ct = ColumnTransformer(transformer, remainder='passthrough') # Creo instancia de la clase. 
X = np.array(ct.fit_transform(X), dtype=np.float) # Aplico la transformación creada a X, que ajusta la transformación a los datos de entrenamiento.
# Por ultimo, el resultado se convierte en un array de tipo float. 



#Regla general: Si las variables NO tiene una cardinalidad u orden, tengo que usar variables dummmy. Si la tienen no las uso. 


# División del dataset en conjunto de entrenamiento y testing
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0) # Se destina el 20% a test y 80% a train.


# Proceso de escalado de variables
# Creamos primero el escalador
sc_X = preprocessing.StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
