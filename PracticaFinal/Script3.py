import cv2
import numpy as np
import os

# Script3.py : El siguiente script se encarga de reconocer a las personas en
# tiempo real. El modelo carga el modelo entrenado en el Script2.py, y 
# luego lo utiliza para predecir quién es la persona que aparece en la cámara.
# Para ello, detecta los rostros en cada frame de la cámara, y para cada rostro
# detectado, lo recorta, lo redimensiona a 150x150 (lo mismo que en el entrenamiento)
# y lo convierte a escala de grises (lo mismo que en el entrenamiento), para luego
# predecir su etiqueta y confianza con el modelo.
# Si la confianza es menor que un umbral, se considera que la persona está reconocida, 
# y se muestra su nombre; si la confianza es mayor que el umbral, se considera que la 
# persona no está registrada, y se muestra "Desconocido". 

# Cargar el modelo entrenado
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPH.xml')

# Mapa etiqueta -> nombre (mismo orden que al entrenar)
dataPath = 'Rostros'
peopleList = sorted(os.listdir(dataPath))  # ordena igual que en entrenamiento
print("Personas registradas:", peopleList)

# Inicializar el detector de rostros y la cámara
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Umbral: por debajo -> reconocido; por encima -> desconocido
# Para LBPH valores típicos: < 70 reconocido, > 70 desconocido
UMBRAL = 70

while True:
    # Leemos un frame de la cámara
    ret, frame = cap.read()
    if not ret: break
    # Espejamos la imagen para que sea más natural al mirarnos en la pantalla
    frame = cv2.flip(frame, 1)
    # Convertimos a escala de grises para detectar los rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Hacemos una copia del frame para recortar los rostros sin modificar el original
    auxFrame = frame.copy()

    # Detectamos los rostros en el frame
    faces = faceClassif.detectMultiScale(gray, 1.3, 5,
                                          minSize=(30,30), maxSize=(200,200))

    # Por cada rostro detectado, lo recortamos, lo redimensionamos a 150x150 y lo predecimos con el modelo
    for (x, y, w, h) in faces:
        # Recortamos el rostro de la imagen original
        rostro = auxFrame[y:y+h, x:x+w]
        # Redimensionamos el rostro a 150x150 píxeles (esto es importante para el modelo)
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
        # Convertimos el rostro a escala de grises para la predicción
        rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)

        # Predecimos la etiqueta y la confianza del rostro detectado
        label, confidence = face_recognizer.predict(rostro_gray)
        
        # Si la confianza es menor que el umbral, lo consideramos reconocido; si no, desconocido
        if confidence < UMBRAL:
            nombre = peopleList[label]
            color = (0, 255, 0)      # verde -> registrado
            texto = f'{nombre} ({confidence:.1f})'
        else:
            nombre = 'Desconocido'
            color = (0, 0, 255)      # rojo -> no registrado
            texto = f'Desconocido ({confidence:.1f})'

        # Dibujamos un rectángulo alrededor del rostro detectado y el texto con el nombre y la confianza
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, texto, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Mostramos el frame con los rectángulos y el texto
    cv2.imshow('Autenticacion', frame)
    if cv2.waitKey(1) == 27:
        break

# Liberamos la cámara y cerramos las ventanas
cap.release()
cv2.destroyAllWindows()