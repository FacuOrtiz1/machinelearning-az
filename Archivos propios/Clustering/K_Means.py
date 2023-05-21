
"""
Created on Tue May 16 19:32:19 2023

@author: Equipo
"""

# K-Means
# ----- Preprocesado de información -----
# Importamos librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


# Importamos el dataset
dataset = pd.read_csv("D:/FACU/machinelearning-az/datasets/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values # Matriz de características 

# Método del codo: averiguar el número óptimo de clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Método del codo")
plt.xlabel("Número de clusters")
plt.ylabel("WCSS(k)")
plt.show()

# Entonces el número óptimo de clusters es k = 5

# Aplicación del método K-Means para segmentar el dataset.
kmeans_opt = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10,random_state=0)
y_kmeans = kmeans_opt.fit_predict(X) # Hace un fit de los datos para ajustar los datos y tambien predice a que cluster pertenece.

# Visuaalización de los clusters 
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0,1],s=100, c="red",label = "Cluster 1")
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1,1],s=100, c="blue",label = "Cluster 2")
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2,1],s=100, c="green",label = "Cluster 3")
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3,1],s=100, c="cyan",label = "Cluster 4")
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4,1],s=100, c="magenta",label = "Cluster 5")
plt.scatter(kmeans_opt.cluster_centers_[:,0] ,kmeans_opt.cluster_centers_[:,1] ,s=300, c="yellow", label = "Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (M$)")
plt.xlabel("Puntuación de gastos (1-100)")
plt.legend()
plt.show()