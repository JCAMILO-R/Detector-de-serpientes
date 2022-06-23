#!/usr/bin/env python
# coding: utf-8

# ## Que es la estimacion de caracteristicas

# cuando hablamos de estimacion de caracteristicas nos vamos a referir aquellas cosas que son de tipo semantico. 
# 
# ejemplo si hablamos de una esfera esto pueden ser el Radio, el Area, su perimetro entre otras caracteristicas

# ## Que es la extraccion de caracteristicas

# es aquel objeto que me genera una característica que no se representa con un valor matemático si no con alguna descripción, por lo cual se le brinda un conjunto de datos o valores lo cuales sean de la mayor utilidad para poderlos interpretar con los algoritmos.

# ### EXPERIMENTO REALIZADO

# 
# 
# 
# 
# 

# 
# 
# ### Paso 1: Importar las librerias

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FastICA, PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ### Paso 2: Importar el conjunto de Base de Datos

# In[2]:


reg = linear_model.LogisticRegression()
archivo = "BDatos3.csv"
ds = pd.read_csv(archivo, sep=';', encoding='latin-1')


# In[3]:


ds


# In[5]:


#Distribucion de clases
print(ds.groupby('# Etiquetas').size())


# In[6]:


#separando los datos en variables dependientes e independientes
X=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values
print(X)
print(y)


# In[7]:


#resumen de los dataset
print("dimensiones de x:")
print(X.shape)
print("dimensiones de y:")
print(y.shape)


# ## Paso 3: dividir el conjunto de datos en el conjunto de entrenamiento y el conjunto de prueba

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#estamos destinando aqui el 20 % para la clasificicacion
#80% para el entrenamiento


# ### proceso para PCA
# #### Paso 4: escalado de características

# In[9]:


sc = StandardScaler() 
x_train = sc.fit_transform(X_train) 
x_test = sc.transform(X_test) 


# #### Paso 5: Aplicación de la función PCA

# In[10]:


pca = PCA(n_components = 2) #valor del parametro
  
x_train = pca.fit_transform(x_train) 
x_test = pca.transform(x_test) 
  
explained_variance = pca.explained_variance_ratio_ 


# #### Paso 6: Ajuste de la regresión logística al conjunto de entrenamiento

# In[11]:


classifier = LogisticRegression(random_state = 0) 
classifier.fit(x_train, y_train) 


# #### Paso 7: predecir el resultado del conjunto de prueba

# In[12]:


# Predecir el resultado del conjunto de prueba usando la función de predicción bajo LogisticRegression
y_pred = classifier.predict(x_test) 


# #### Paso 8: hacer la matriz de confusión

# In[14]:


# haciendo una matriz de confusión entre el conjunto de prueba de Y y el valor predicho.
#resumen de las predicciones hechas por el clasificador
from sklearn.linear_model import LogisticRegression
print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

#precision
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# ### Paso 9: predecir el resultado del conjunto de entrenamiento

# In[15]:


# Predicting the training set 
# result through scatter plot  
from matplotlib.colors import ListedColormap 
  
X_set, y_set = x_train, y_train 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                     stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1, 
                     stop = X_set[:, 1].max() + 1, step = 0.01)) 
  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('yellow', 'white', 'aquamarine'))) 
  
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
  
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j) 
  
plt.title('Logistic Regression (Training set)') 
plt.xlabel('PC1') # for Xlabel 
plt.ylabel('PC2') # for Ylabel 
plt.legend() # to show legend 
  
# show scatter plot 
plt.show() 


# ### Paso 10: visualizar los resultados del conjunto de pruebas

# In[16]:


# Visualising the Test set results through scatter plot 
from matplotlib.colors import ListedColormap 
  
X_set, y_set = x_test, y_test 
  
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                     stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1, 
                     stop = X_set[:, 1].max() + 1, step = 0.01)) 
  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))  
  
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
  
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j) 
  
# title for scatter plot 
plt.title('Logistic Regression (Test set)')  
plt.xlabel('PC1') # for Xlabel 
plt.ylabel('PC2') # for Ylabel 
plt.legend() 
  
# show scatter plot 
print(cm)
plt.show() 


# ### Proceso LDA
# 
# #### Paso 4: escalado de características

# In[17]:


sc = StandardScaler()
x1_train = sc.fit_transform(X_train)
x1_test = sc.transform(X_test)


# #### Paso 5: Aplicación de la función LDA

# In[18]:


lda = LDA(n_components=2)
x1_train = lda.fit_transform(x1_train, y_train)
x1_test = lda.transform(x1_test)


# #### Paso 6: Ajuste de la regresión logística al conjunto de entrenamiento

# In[19]:


classifier = LogisticRegression(random_state = 0) 
classifier.fit(x1_train, y_train)


# #### Paso 7: predecir el resultado del conjunto de prueba

# In[20]:


y_pred = classifier.predict(x1_test)


# #### Paso 8: hacer la matriz de confusión

# In[21]:


cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))


# ### Paso 10: visualizar los resultados del conjunto de pruebas

# In[22]:


# Visualising the Test set results through scatter plot 
x_test=x1_test 
from matplotlib.colors import ListedColormap 
  
X_set, y_set = x_test, y_test 
  
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                     stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1, 
                     stop = X_set[:, 1].max() + 1, step = 0.01)) 
  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))  
  
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
  
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j) 
  
# title for scatter plot 
plt.title('Logistic Regression (Test set)')  
plt.xlabel('LDA1') # for Xlabel 
plt.ylabel('LDA2') # for Ylabel 
plt.legend() 
  
# show scatter plot 
print(cm)
plt.show() 


# ### CLASIFICADOR

# In[23]:


archivo = "BDatos3.csv"
ds = pd.read_csv(archivo, sep=';', encoding='latin-1')
x=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values
#Dividiendo el dataset en entrenamiento y prueba
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# #### k vecinos mas cercanos 

# In[25]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
#resumen de las predicciones hechas por el clasificador
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
#precision
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[26]:


archivo = "BDatos3.csv"
ds = pd.read_csv(archivo, sep=';', encoding='latin-1')
x=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values
#Dividiendo el dataset en entrenamiento y prueba
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# #### Regresión logística

# In[28]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x1_train = sc.fit_transform(X_train)
x1_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) 
classifier.fit(x1_train, y_train)
y_pred = classifier.predict(x1_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))


# # Conclusiones
# 
# 
# - Al no realizar un preprocesamiento en las muestras antes de conformar la base de datos se presentan similitudes en algunas características de las diferentes serpientes lo cual afecta su clasificación y el entrenamiento del sistema
# - Si un clasificador presentar un mayor porcentaje de precisión al momento de realizar la clasificación no implica que este método de clasificación sea mejor, esto solo nos dice que se adapta mejor al problema que hemos planteado o a nuestra base de datos. De manera similar ocurre cuando utilizamos diferentes extractores de características con el mismo clasificador
# 
# 

# # REFERENCIAS

# * https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/
# * https://www.geeksforgeeks.org/principal-component-analysis-with-python/
# * https://stackoverflow.com/questions/11283220/memory-error-in-python
# * https://scikit-learn.org/stable/modules/decomposition.html#ica
# * https://www.youtube.com/watch?v=wTA8nE-BJGc
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




