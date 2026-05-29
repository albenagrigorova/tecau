import cv2
import numpy as np
import os

# Script2.py : El siguiente script se encarga de entrenar el modelo 
# de reconocimiento facial con las imágenes capturadas en el Script1.py.
# Para ello, recorre las carpetas de rostros, etiqueta cada imagen con el 
# número de persona correspondiente, y entrena un modelo LBPH (Local Binary 
# Patterns Histograms) con esas imágenes y etiquetas (podríamos usar otros pero
# tardan la vida y media en entrenarse, y no son tan eficientes). Finalmente, 
# guarda el modelo entrenado en un archivo XML para su uso posterior en el 
# reconocimiento facial en tiempo real o en otras aplicaciones.

# 1. Etiquetado de imágenes
dataPath = 'Rostros'
peopleList = os.listdir(dataPath)

labels = []
facesData = []
label = 0

# Recorremos cada carpeta (persona) y cada imagen dentro de esa carpeta
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    for fileName in os.listdir(personPath):
        # Cargamos la imagen en escala de grises (esto es importante para el entrenamiento)
        img = cv2.imread(personPath + '/' + fileName, 0)  # escala de grises
        if img is not None:
            # Agregamos la imagen a facesData y la etiqueta (número de persona) a labels
            facesData.append(img)
            labels.append(label)
    label += 1

# 2. Entrenamiento — hay otros como EigenFaces o FisherFaces, 
# pero LBPH suele ser más robusto con pocas imágenes
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenamos el modelo con las imágenes y sus etiquetas correspondientes
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))
face_recognizer.write('modeloLBPH.xml')
print("Modelo guardado.")