"""
Created on Tue May 16 19:32:19 2023

@author: Equipo
"""

# Clustering jerarquico

# ----- Preprocesado de información -----
# Importamos librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Importamos el dataset
dataset = pd.read_csv("D:/FACU/machinelearning-az/datasets/Part 4 - Clustering/Section 25 - Hierarchical Clustering/Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values # Matriz de características 

# Dendrograma: Se busca el número óptimo de clusters. 
dendrograma = sch.dendrogram(sch.linkage(X,method="ward")) # linkage indica que es el clustering conglomerativo 
# El método ward es un método que intenta minimizar la varianza que existe entre los puntos de los clusters. 
# En K-Means un método que usabamos era intentar minimazar la suma de los cuadradados de los puntos internos con respecto al baricentro. 
# Acá se intenta minizar la varianza entre los puntos del cluster. 
plt.figure(1)
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancias Euclidea entre los clusters")
plt.show()

# Entonces el número óptimo de clusters es k = 5

# Aplicación del clustering jerarquico para segmentar el dataset.
hc = AgglomerativeClustering(n_clusters=5, linkage="ward", affinity="deprecated")
y_hc = hc.fit_predict(X) 


# Visuaalización de los clusters 
plt.figure(2)
plt.scatter(X[y_hc==0, 0], X[y_hc==0,1],s=100, c="red",label = "Cluster 1")
plt.scatter(X[y_hc==1, 0], X[y_hc==1,1],s=100, c="blue",label = "Cluster 2")
plt.scatter(X[y_hc==2, 0], X[y_hc==2,1],s=100, c="green",label = "Cluster 3")
plt.scatter(X[y_hc==3, 0], X[y_hc==3,1],s=100, c="cyan",label = "Cluster 4")
plt.scatter(X[y_hc==4, 0], X[y_hc==4,1],s=100, c="magenta",label = "Cluster 5")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (M$)")
plt.xlabel("Puntuación de gastos (1-100)")
plt.legend()
plt.show()