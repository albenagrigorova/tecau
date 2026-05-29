import cv2
import os
import sys

# Script1.py : El siguiente script se encarga de sacar las imágenes de la 
# persona a la que queremos reconocer. Para ello, se abre la cámara y 
# se detecta su rostro, sacando fotos de pantalla de cada segundo que
# se detecta un rostro, hasta llegar a un número de imágenes determinado,
# el cual puede variar gracias a añadir por argumentos el número y el
# nombre deseado de la persona.
# Un ejemplo sería > python Script1.py 600 "Carla"

# Primero, escogemos el número de imágenes a capturar y el nombre de la persona 
# (opcionalmente por argumentos)
n_imagenes = int(sys.argv[1]) if len(sys.argv) > 1 else 300
nombre_persona = sys.argv[2] if len(sys.argv) > 2 else "persona"

# Se crea la carpeta de la persona si no existe
carpeta = f'Rostros/{nombre_persona}'
if not os.path.exists(carpeta):
    os.makedirs(carpeta)

# Inicializamos la cámara y el detector de rostros(tenemos que instalar 
# opencv-contrib-python, que con el opencv-python solamente no tenemos el 
# haarcascade_frontalface_default.xml)
cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Averiguar cuántas imágenes hay ya en la carpeta (esto es para no sobreescribirlas, 
# y para poder sacar más en caso de querer seguir capturando después)
archivos_existentes = len(os.listdir(carpeta))

# Inicializar los contadores (empiezan desde el último número de imagen)
count = archivos_existentes
capturadas_ahora = 0         

# El ciclo principal de captura
while True:
    # Leemos un frame de la cámara
    ret, frame = cap.read()
    # cv2.flip sirve para espejar la imagen, así es más natural al mirarnos en la pantalla
    frame = cv2.flip(frame, 1)
    # Convertimos a escala de grises para detectar los rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Hacemos una copia del frame para recortar los rostros sin modificar el original
    auxFrame = frame.copy()
    # Detectamos los rostros en el frame 
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    # Por cada rostro detectado, lo recortamos, lo redimensionamos a 150x150 y lo guardamos en la carpeta
    for (x, y, w, h) in faces:
        # Dibujamos un rectángulo alrededor del rostro detectado
        cv2.rectangle(frame, (x,y), (x+w,y+h), (128,0,255), 2)
        # Recortamos el rostro de la imagen original (no del gris, para que se guarde a color)
        rostro = auxFrame[y:y+h, x:x+w]
        # Redimensionamos el rostro a 150x150 píxeles (esto es importante para el entrenamiento posterior)
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
        # Guarda la imagen con el número global
        cv2.imwrite(f'{carpeta}/rostro_{count}.jpg', rostro)
        
        # Incrementamos el contador global y el contador de esta sesión
        count += 1
        capturadas_ahora += 1

    # Mostramos información útil en la pantalla (esto se puede eliminar pero para saber cuantas
    # imágenes se han capturado y cuantas hay en total viene muy bien)
    texto = f'Capturadas hoy: {capturadas_ahora}/{n_imagenes} | Total carpeta: {count}'
    cv2.putText(frame, texto, (10,20), 2, 0.6, (128,0,255), 1, cv2.LINE_AA)
    # Mostramos el frame con los rectángulos y el texto
    cv2.imshow('frame', frame)

    # El ciclo se rompe cuando las capturadas en ESTA sesión llegan a N
    if capturadas_ahora >= n_imagenes or cv2.waitKey(1) == 27:
        break

# Liberamos la cámara y cerramos las ventanas
cap.release()
cv2.destroyAllWindows()