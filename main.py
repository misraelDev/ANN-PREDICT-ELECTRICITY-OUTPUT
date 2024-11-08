# 1 PREPROCESAMIENTO DE DATOS

import numpy as np
import pandas as pd
import tensorflow as tf

# IMPORTACIION DE DATOS

# Cargar el conjunto de datos desde un archivo de Excel
dataset = pd.read_excel('Folds5x2_pp.xlsx')

# Extraer las variables de características (todas las columnas excepto la última)
X = dataset.iloc[:, :-1].values

# Extraer la variable objetivo (la última columna)
y = dataset.iloc[:, -1].values

print(X)
print(y)

# DIVIDIR EL CONJUNTO DE DATOS EN EL CONJUNTO DE ENTRENAMIENTO Y EL CONJUNTO DE PRUEBA
# Funcion que tomara como entrada el conjunto de datos y devolvera el modelo entrenado en x, y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 2 CONSTRUIR LA RED NEURONAL ARTIFICIAL

ann = tf.keras.models.Sequential()

# Agregar la capa de entrada y la primera capa oculta
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adición de la segunda capa oculta
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adición de la capa de salida
ann.add(tf.keras.layers.Dense(units=1))

# 3 ENTRENAR LA RED NEURONAL ARTIFICIAL

# Compilar la red neuronal
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Entrenamiento del modelo ANN en el conjunto de entrenamiento
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
