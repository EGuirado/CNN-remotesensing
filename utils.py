import os
import sys
import numpy as np
import cv2
from keras.utils import np_utils

# Función para cargar las imágenes para utilizarlas con Keras para ResNet y
# DenseNet teniéndolas en el formato necesario para GoogLeNet.
def loadData(image_dir, img_width, img_height, classes):
    sub_dirs = next(os.walk(image_dir))[1]
    if len(sub_dirs) == 0:
        sys.exit("No subfolders found")

    img_paths = []
    labels = []
    for sub_dir in sub_dirs:
        files = os.listdir(image_dir + "/" + sub_dir)
        if len(files) == 0:
            print("No images found in " + sub_dir)
            sys.exit(1)
        for f in files:
            if (f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg')
                or f.endswith('.JPEG')):
                img_paths.append(os.path.join(image_dir, sub_dir, f))
                labels.append(sub_dir)

    images = np.array([cv2.resize((cv2.imread(img)).astype(np.float32),
                        (img_width, img_height)) for img in img_paths])

    labels = np.array(labels)
    for i in range(len(classes)):
        labels[labels == classes[i]] = i
    labels = np_utils.to_categorical(labels, len(classes))

    return images, labels


# Función para obtener la salida del modelo sin la última capa,
# a lo que vamos a llamar características
def getFeatures(model, images, batch_size):
    return model.predict(images, batch_size = batch_size)


# Función para calcular el accuracy dados la matriz dispersa de etiquetas
# y la matriz de predicciones conteniendo probabilidades
def getAccuracy(labels, preds):
    vec_labels = np.argmax(labels, axis = 1)
    vec_preds = np.argmax(preds, axis = 1)
    return sum(vec_labels == vec_preds)/len(vec_labels)
