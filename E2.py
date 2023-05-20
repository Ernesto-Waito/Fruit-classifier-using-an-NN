import os
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.clear_session()

data_entrenamiento = './dataset/train'
data_validacion = './dataset/validation'

# Parámetros
epocas = 20
altura, longitud = 100, 100
batch_size = 32
pasos = 1000
pasos_validacion = 200
filtrosC1 = 32
filtrosC2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 36
lr = 0.0005

# Preprocesamiento de imágenes
entrenamiento_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

validacion_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Carga de imágenes
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

# Creación del modelo CNN
cnn = keras.Sequential()

cnn.add(keras.layers.Conv2D(filtrosC1, tamano_filtro1, padding='same', input_shape=(altura, longitud, 3), activation='relu'))
cnn.add(keras.layers.MaxPooling2D(pool_size=tamano_pool))

cnn.add(keras.layers.Conv2D(filtrosC2, tamano_filtro2, padding='same', activation='relu'))
cnn.add(keras.layers.MaxPooling2D(pool_size=tamano_pool))

cnn.add(keras.layers.Flatten())
cnn.add(keras.layers.Dense(256, activation='relu'))
cnn.add(keras.layers.Dropout(0.5))
cnn.add(keras.layers.Dense(clases, activation='softmax'))

# Compilación del modelo
cnn.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=lr), metrics=['accuracy'])

# Entrenamiento del modelo
cnn.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion)

# Guardar el modelo y los pesos
dir = './modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)

cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')
