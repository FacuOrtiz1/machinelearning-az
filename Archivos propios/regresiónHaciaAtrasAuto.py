# -*- coding: utf-8 -*-
"""
Created on Sat May  6 17:23:01 2023

@author: Equipo
"""

import numpy as np
import statsmodels.api as sm
# Eliminación hacia atrás utilizando solamente p-valores:
def backwardElimination(x, y, SL):    
    numVars = len(x[0]) # Calculo el número de variables de la matriz = 6
    for i in range(0, numVars): #[0,1,2,3,5]       
        regressor_OLS = sm.OLS(y, x.tolist()).fit() #Creo el regresor con todas las variables   
        maxVar = max(regressor_OLS.pvalues).astype(float)   # obtengo el mayor pvalue     
        if maxVar > SL:            #Consulto si es mayor que el SL tolerado
            for j in range(0, numVars - i):      # depende dinámicamente de i           
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):       #Chequeo si el maximo es ese valor             
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary() 
    return x 
 
# Eliminación hacia atrás utilizando  p-valores y el valor de  R Cuadrado Ajustado:
def backwardElimination_R(x, y, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 