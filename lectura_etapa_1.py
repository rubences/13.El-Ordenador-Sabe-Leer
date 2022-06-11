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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


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

#Se libera la webCam y se destruyen las ventanas
webCam.release()
cv2.destroyAllWindows()