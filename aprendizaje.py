#-----------------------------------------------------------------------------------------
#
# Modulos necesarios:
#   TENSORFLOW 1.13.1
#   KERAS 2.2.4
#   OPENCV 3.4.5.20
#   PYTTSX3 2.7.1
#   SCIKIT-LEARN 0.21.1
#   NUMPY 1.16.3
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------


from __future__ import print_function
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

from mnist import MNIST


#Carga de las imágenes
emnist_data = MNIST(path='datas\\', return_type='numpy')
emnist_data.select_emnist('letters')
Imagenes, Etiquetas = emnist_data.load_training()


print("Cantidad de imágenes ="+str(len(Imagenes)))
print("Cantidad de etiquetas ="+str(len(Etiquetas)))



#Conversión de las imágenes y etiquetas en tabla numpy
import numpy as np
Imagenes = np.asarray(Imagenes)
Etiquetas = np.asarray(Etiquetas)


#Dimensión de las imégenes de trabajo y de aprendizaje
largoImagen = 28
anchoImagen = 28


#Las imágenes están en la forma de una tabla de 124800 líneas y 784 columnas
#Las transformamos en una tabla que contiene 124800 líneas que contiene una tabla de 28*28 columnas
print("Transformación de las tablas de imágenes...")
Imagenes = Imagenes.reshape(124800, anchoImagen, largoImagen)
Etiquetas= Etiquetas.reshape(124800, 1)

print("Visualización de la imagen N.° 70000...")
from matplotlib import pyplot as plt
plt.imshow(Imagenes[70000])
plt.show()

print(Etiquetas[70000])

#En informática, los índices de las listas deben empezar por cero...")
Etiquetas = Etiquetas-1


print("Etiqueta de la imagen N.° 70000...")
print(Etiquetas[70000])



#Creación de los conjutnos de aprendizaje y de prueba
imagenes_aprendizaje, imagenes_validacion, etiquetas_aprendizaje, etiquetas_validacion = train_test_split(Imagenes, Etiquetas, test_size=0.25, random_state=42)

#Adición de un tercer valor a nuestras tablas de imágenes para que puedan ser utilizadas por la red neuronal, especialmente el parámetro input_shape de la función Conv2D
imagenes_aprendizaje = imagenes_aprendizaje.reshape(imagenes_aprendizaje.shape[0], anchoImagen, largoImagen, 1)
print(imagenes_aprendizaje.shape)

imagenes_validacion = imagenes_validacion.reshape(imagenes_validacion.shape[0], anchoImagen, largoImagen, 1)

#Creación de una variable que sirve de imagen de trabajo a la red neuronal
imagenTrabajo = (anchoImagen, largoImagen, 1)

#Adaptación a la escala
imagenes_aprendizaje = imagenes_aprendizaje.astype('float32')/255
imagenes_validacion = imagenes_validacion.astype('float32')/255

# Creación de las categorías en un sistema de codificación One-Hot
cantidad_de_clases = 26
etiquetas_aprendizaje = keras.utils.to_categorical(etiquetas_aprendizaje, cantidad_de_clases)
etiquetas_validacion = keras.utils.to_categorical(etiquetas_validacion, cantidad_de_clases)

# Red neuronal convolucional
# 32 filtros de dimensiones 3x3 con una función de activación de tipo RELU
# El filtro tiene en la entrada la imagen de trabajo
redCNN = Sequential()
redCNN.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=imagenTrabajo))

#Una segunda capa de 64 filtros de dimensión 3x3
redCNN.add(Conv2D(64, (3, 3), activation='relu'))

#Una función de pooling
redCNN.add(MaxPooling2D(pool_size=(2, 2)))
redCNN.add(Dropout(0.25))

#Un aplanado
redCNN.add(Flatten())

#La red neuronal con 128 neuronas en la entrada
#una función de activación de tipo ReLU
redCNN.add(Dense(128, activation='relu'))
redCNN.add(Dropout(0.5))

#Una última capa de tipo softmax
redCNN.add(Dense(cantidad_de_clases, activation='softmax'))

#Compilación del modelo
redCNN.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# Aprendizaje con una fase de validación
# en los conjuntos de prueba
batch_size = 128
epochs = 10

redCNN.fit(imagenes_aprendizaje, etiquetas_aprendizaje,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(imagenes_validacion, etiquetas_validacion))



# Guardado del modelo
redCNN.save('modelo/modelo_caso_practicov2.h5')

# Evaluación de la precisión del modelo
score = redCNN.evaluate(imagenes_validacion, etiquetas_validacion, verbose=0)
print('Precisión en los datos datos de validación:', score[1])