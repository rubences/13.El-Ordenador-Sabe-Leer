#-----------------------------------------------------------------------------------------
#
# Módulos necesarios:
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


import cv2
import numpy as np

#Module de paroles
import pyttsx3 as pyttsx

#Module Keras permettant l'utilisation de notre réseau de neurones
from keras.models import load_model

#Module de gestion des processus
import threading


#De manera predeterminada se activa la lectura de letra en voz alta
lectureActivee = True

#Tiempo de espera en segundos entre cada lectura de letra en voz alta
duraciónDesactivacionLecturaDeLetra = 5

#función de reactivación de la lectura de letra en voz alta
def activacionLectura():
    print('Activación de la lectura de letras')
    global lecturaActivada
    lecturaActivada=True

#dimensiones de la zona de escritura
zonaEscrituraLargoMin = 540
zonaEscrituraLargoMax = 590
zonaEscrituraAnchoMin = 300
zonaEscrituraAnchoMax = 340

#Initialización de la voz
print('Initialización de la voz')
engine = pyttsx.init()

#Elección de la voz en español
voice = engine.getProperty('voices')[0]
engine.setProperty('voice', voice.id)

#prueba de la voz
engine.say('Modo lectura de letras activado')
engine.runAndWait()

print('Inicialización del modelo de aprendizaje')

#Carga del modelo entrenado
cnn_model = load_model('modelo/modelo_caso_practicoV2.h5')
kernel = np.ones((5, 5), np.uint8)

#Tabla de letras con su número
letras = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
           11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T',
           21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: '-'}



#De manera predeterminada se elige que se detecte la letra Z
prediccion = 26

#De manera predeterminada no se hace ninguna predicción.
letraPredicha = False


print('Inicialización de la webcam')
webCam = cv2.VideoCapture(0)
if webCam.isOpened():
    largoWebcam = webCam.get(3)
    anchoWebcam = webCam.get(4)
    print('Resolución:' + str(largoWebcam) + " X " + str(anchoWebcam))
else:
    print('ERROR')

while True:

    #De manera predeterminada no se hace ninguna detección.
    letraPredicha = False

    # Captura de la imagen en la variable Frame
    # La variable lecturaOK es igual a True si la función read() está operativa
    (lecturaOK, frame) = webCam.read()

    (grabbed, frame) = webCam.read()
    tsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contornos_canny = cv2.Canny(gris, 30, 200)

    contornos = cv2.findContours(contornos_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, 0.012 * perimetro, True)
        x, y, w, h = cv2.boundingRect(approx)

        #Se encuadra la zona de escritura en función de los parámetros de largo y ancho de la pizarra
        if len(approx) == 4 and h>zonaEscrituraAnchoMin and w>zonaEscrituraLargoMin and h<zonaEscrituraAnchoMax and w<zonaEscrituraLargoMax:

            #Encuadre de la zona de escritura
            area = cv2.contornoArea(contorno)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3);

            # Captura de la imagen a partir de la zona de escritura con un margen interior (padding) de 10
            # píxeles para aislar solo la letra
            letra = gris[y + 10:y + h - 10, x + 10:x + w - 10]

            # Se detectan los contornos de la letra con la ayuda del algoritmo de Canny
            cannyLetra = cv2.Canny(letra, 30, 200)
            contornosLetra = cv2.findContornos(cannyLetra.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

            # Si hay una letra d dibujada
            if len(contornosLetra) > 5:

                # Creación de una tabla para el almacenamiento de la imagen de la letra
                captureAlphabetTMP = np.zeros((400, 400), dtype=np.uint8)

                # Se detecta el contorno más grande (Reverse = True)
                cnt = sorted(contornosLetra, key=cv2.contornoArea, reverse=True)[0]

                # Se guardan las coordenadas del rectángulo de delimtación de la letra
                xc, yc, wc, hc = cv2.boundingRect(cnt)


                for contornoLetra in contornosLetra:
                    area = cv2.contourArea(contorno)
                    if area > 1000:

                        # Se dibujan los contornos de la letra para una lectura mejor (Trazo de 10 px)
                        cv2.drawContours(captureAlphabetTMP, contornoLetra, -1, (255, 255, 255), 10)

                        # Se captura la letra y se guardan los valores de los píxeles de la zona capturada en una tabla
                        capturaLetra = np.zeros((400, 400), dtype=np.uint8)
                        capturaLetra = captureAlphabetTMP[yc:yc + hc, xc:xc + wc]

                        # Se pueden capturar sombras en la zona de escritura provocando errores de
                        # reconocimiento. Si se dectecta una sombra, una de las dimensiones de la tabla de captura es
                        # igual a cero porque no se ha detectado ningún contorno de letra
                        visualizacionLetraCapturada = True
                        if (capturaLetra.shape[0] == 0 or capturaLetra.shape[1] == 0):
                            print("¡ERROR A CAUSA DE LAS SOMBRAS!: ")
                            visualizacionLetraCapturada = False

                        #Si no es una sombra, se muestra la letra capturada en la pantalla
                        if visualizacionLetraCapturada:
                            cv2.destroyWindow("ContornosLetra");
                            cv2.imshow("ContornosLetra", capturaLetra)

                            # Redimensionamiento de la imagen
                            newImage = cv2.resize(captureLettre, (28, 28))
                            newImage = np.array(newImage)
                            newImage = newImage.astype('float32') / 255
                            newImage.reshape(1, 28, 28, 1)

                            # Realizatión de la predicción
                            prediccion = cnn_model.predict(newImage.reshape(1, 28, 28,1))[0]
                            prediccion = np.argmax(prediccion)

                            # Se indica que se ha detectado una letra
                            letraPredicha = True


                if letraPredicha:

                    #Se desactiva la lectura de letras en voz alta
                    print('Desactivación de la lecture de letra ' + str(duracionDesactivacionLecturaDeLetra) + " segundos")
                    lectureActivee = False

                    #Se muestra el número de la letra predicho
                    #Se añade +1 porque la primera letra del alfabeto tiene valor 0 en nuestro sistema de predicción
                    #Entonces tiene el valor 1 en nuestra tabla de correspondencia
                    print("Detección:" + str(letraPredicha))
                    print("Predicción = " + str(prediccion))

                    #Lectura en voz alta de la letra predicha
                    if (letraPredicha and prediccion != 26):
                        engine.say('Leo la letra ' + str(letras[int(prediccion) + 1]))
                        engine.runAndWait()
                        letraPredicha = False

                    if (letraPredicha and prediccion == 26):
                        engine.say('No comprendo la letra escrita')
                        engine.runAndWait()
                        letraPredicha = False

                    #Pausa del proceso de lectura de la letra y luego llama a la funciónn para la reactivación de la
                    #lectura
                    timer = threading.Timer(duracionDesactivacionLecturaDeLetra, activacionLectura)
                    timer.start()


    # Visualización de la imagen capturada por la webcam
    cv2.imshow("IMAGEN", frame)
    cv2.imshow("HSV", tsv)
    cv2.imshow("GRIS", gris)
    cv2.imshow("CANNY", contornos_canny)

    # Condición de salida del bucle While
    # > Tecla Escape para salir
    key = cv2.waitKey(1)
    if key == 27:
        break

#Se libera la webCam y se destruyen todas las ventanas
webCam.release()
cv2.destroyAllWindows()