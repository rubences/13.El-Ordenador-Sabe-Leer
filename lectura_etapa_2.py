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

#dimensiones de la pizarra
zonaEscrituraLargoMin = 540
zonaEscrituraLargoMax = 590
zonaEscrituraAnchoMin = 300
zonaEscrituraAnchoMax = 340


print('Inicialización de la webcam')
webCam = cv2.VideoCapture(0)
if webCam.isOpened():
    largoWebcam = webCam.get(3)
    anchoWebcam = webCam.get(4)
    print('Resolución:' + str(largoWebcam) + " X " + str(anchoWebcam))
else:
    print('ERROR')

while True:

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

            # Se detectan los contornos de la letra con la ayuda del algoritmo Canny
            cannyLetra = cv2.Canny(letra, 30, 200)
            contornosLetra = cv2.findContornos(cannyLetra.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

            # Si hay una letra d dibujada
            if len(contornosLetra) > 5:

                # Creación de una tabla para el almacenamiento de la imagen de la letra
                capturaAlphabetTMP = np.zeros((400, 400), dtype=np.uint8)

                # Se detecta el contorno más grande (Reverse = True)
                cnt = sorted(contoursLettre, key=cv2.contourArea, reverse=True)[0]

                # Se guardan las coordenadas del rectángulo de delimitación de la letra
                xc, yc, wc, hc = cv2.boundingRect(cnt)


                for contornoLetra in contornosLetra:
                    area = cv2.contornoArea(contorno)
                    if area > 1000:

                        # Se dibujan los contornos de la letra para una lectura mejor (Trazo de 10 px)
                        cv2.drawContours(captureAlphabetTMP, contornoLetra, -1, (255, 255, 255), 10)

                        # Se captura la letra y se guardan los valores de los píxeles de la zona capturada en una tabla
                        capturaLetra = np.zeros((400, 400), dtype=np.uint8)
                        capturaLetra = captureAlphabetTMP[yc:yc + hc, xc:xc + wc]


                        #Se pueden capturar sombras en la zona de escritura provocando errores de
                        #reconocimiento. Si se dectecta una sombra, una de las dimensiones de la tabla de captura es
                        #igual a cero porque no se ha detectado ningún contorno de letra
                        visualizaciónLetraCapturada = True
                        if (capturaLetra.shape[0] == 0 or capturaLetra.shape[1] == 0):
                            print("¡ERROR A CAUSA DE LAS SOMBRAS!: ")
                            visualizaciónLetraCapturada = False

                        #Si no es una sombra, se muestra la letra capturada en la pantalla
                        if visualizaciónLetraCapturada:
                            cv2.destroyWindow("ContornosLetra");
                            cv2.imshow("ContornosLetra", capturaLetra)

                            # Redimensionamiento de la imagen
                            newImage = cv2.resize(capturaLetra, (28, 28))
                            newImage = np.array(newImage)
                            newImage = newImage.astype('float32') / 255
                            newImage.reshape(1, 28, 28, 1)



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

#Se libera la webcam y se destruyen todas las ventanas
webCam.release()
cv2.destroyAllWindows()