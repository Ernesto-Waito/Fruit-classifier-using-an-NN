import numpy as np
import tensorflow as tf
from tensorflow import keras

longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'

cnn = keras.models.load_model(modelo)
cnn.load_weights(pesos)

# Función que recibirá la imagen y dirá lo que es
def predict(file):
    x = keras.preprocessing.image.load_img(file, target_size=(longitud, altura))
    x = keras.preprocessing.image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    arreglo = cnn.predict(x)
    resultado = arreglo[0]
    respuesta = np.argmax(resultado)

    # Mostrar respuesta
    if respuesta == 0:
        print('perro')
    elif respuesta == 1:
        print('1')
    elif respuesta == 2:
        print('2')
    elif respuesta == 3:
        print('3')
    elif respuesta == 4:
        print('4')
    elif respuesta == 5:
        print('5')
    elif respuesta == 6:
        print('6')
    elif respuesta == 7:
        print('7')
    elif respuesta == 8:
        print('8')
    elif respuesta == 9:
        print('9')
    elif respuesta == 10:
        print('10')
    elif respuesta == 11:
        print('11')
    elif respuesta == 12:
        print('12')
    elif respuesta == 13:
        print('13')
    elif respuesta == 14:
        print('14')
    elif respuesta == 15:
        print('15')
    elif respuesta == 16:
        print('16')
    elif respuesta == 17:
        print('17')
    elif respuesta == 18:
        print('18')
    elif respuesta == 19:
        print('19')
    elif respuesta == 20:
        print('20')
    elif respuesta == 21:
        print('21')
    elif respuesta == 22:
        print('22')
    elif respuesta == 23:
        print('23')
    elif respuesta == 24:
        print('24')
    elif respuesta == 25:
        print('25')
    elif respuesta == 26:
        print('26')
    elif respuesta == 27:
        print('27')
    elif respuesta == 28:
        print('28')
    elif respuesta == 29:
        print('29')
    elif respuesta == 30:
        print('30')
    elif respuesta == 31:
        print('31')
    elif respuesta == 32:
        print('32')
    elif respuesta == 33:
        print('33')
    elif respuesta == 34:
        print('34')
    elif respuesta == 35:
        print('35')

# Ejemplo de uso
imagen = 'ruta/a/la/imagen.jpg'
predict(imagen)


