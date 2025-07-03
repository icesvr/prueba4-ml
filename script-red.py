import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, ConfusionMatrixDisplay)
from tensorflow.keras.utils import to_categorical, set_random_seed
from tensorflow.keras import layers, models
from tensorflow.keras import Input, Model
import tensorflow as tf
import tf2onnx

# Configuración global
set_random_seed(812)
np.random.seed(812)

CLASSES = ['fresa', 'naranja', 'pera', 'tomate', 'platano']
IMG_SIZE = (128, 128)
NUM_CLASSES = len(CLASSES)
FOLDER_PATH = 'dataset/'


def cargar_datos(folder_path):
    image_list, labels = [], []
    for filename in os.listdir(folder_path):
        label = obtener_etiqueta_desde_nombre(filename)
        if label is not None and (filename.endswith('.jpg') or filename.endswith('.png')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            img_array = np.array(img) / 255.0  # Normaliza a 0–1
            image_list.append(img_array)
            labels.append(label)
    return np.array(image_list), np.array(labels)


def obtener_etiqueta_desde_nombre(filename):
    for idx, clase in enumerate(CLASSES):
        if clase in filename.lower():
            return idx
    return None


#def construir_modelo(input_shape, num_classes):
 #   model = models.Sequential([
 #       layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
 #       layers.MaxPooling2D((2, 2)),
 #       layers.Conv2D(64, (3, 3), activation='relu'),
 #       layers.MaxPooling2D((2, 2)),
 #       layers.Flatten(),
 #       layers.Dense(128, activation='relu'),
 #       layers.Dense(num_classes, activation='softmax')
  #  ])
 #   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 #   return model
def construir_modelo(input_shape, num_classes):
    inputs = Input(shape=input_shape, name="input")
    x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def entrenar_modelo(model, X_train, y_train, epochs=70, batch_size=16):
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)


def evaluar_modelo(model, X_test, y_test):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    print("Exactitud:", acc)
    print("Precisión:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot()
    plt.show()


def mostrar_imagen(X, index):
    img = (X[index] * 255).astype(np.uint8)
    plt.imshow(Image.fromarray(img))
    plt.axis('off')
    plt.show()

def predecir_imagen(ruta_imagen, model):
    img = Image.open(ruta_imagen).convert('RGB').resize(IMG_SIZE)  # ← Abre y redimensiona
    img_array = np.array(img) / 255.0                              # ← Convierte a array y normaliza
    img_array = np.expand_dims(img_array, axis=0)                 # ← Le agrega dimensión de batch

    prediccion = model.predict(img_array)                         # ← Aquí se usa img_array
    clase_idx = np.argmax(prediccion)
    clase_nombre = CLASSES[clase_idx]

    print(f"Predicción: {clase_nombre} (confianza: {prediccion[0][clase_idx]:.2f})")

    plt.imshow(img)
    plt.title(f"Predicción: {clase_nombre}")
    plt.axis('off')
    plt.show()


def validar_predicciones(model, ruta_img1, ruta_img2, clases, img_size=(128, 128)):
    def cargar_y_preprocesar(ruta):
        img = Image.open(ruta).convert('RGB').resize(img_size)
        img_array = np.array(img) / 255.0  # Normaliza entre 0 y 1
        return np.expand_dims(img_array, axis=0)  # Agrega dimensión de batch

    img1 = cargar_y_preprocesar(ruta_img1)
    img2 = cargar_y_preprocesar(ruta_img2)

    pred1 = model.predict(img1)
    pred2 = model.predict(img2)

    clase1 = clases[np.argmax(pred1)]
    clase2 = clases[np.argmax(pred2)]

    print(f"\n--- Predicción 1: {ruta_img1} ---")
    print("Vector de salida:", pred1[0])
    print("Clase predicha:", clase1)

    print(f"\n--- Predicción 2: {ruta_img2} ---")
    print("Vector de salida:", pred2[0])
    print("Clase predicha:", clase2)

def exportar_a_onnx(model, input_shape, nombre_salida):
    spec = (tf.TensorSpec([None, *input_shape], tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=nombre_salida)
    print(f"Modelo exportado a ONNX en: {nombre_salida}")

# ========= MAIN =========
if __name__ == '__main__':
    X, y = cargar_datos(FOLDER_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    model = construir_modelo((IMG_SIZE[0], IMG_SIZE[1], 3), NUM_CLASSES)
    ajuste = entrenar_modelo(model, X_train, y_train)

    evaluar_modelo(model, X_test, y_test)
    # mostrar una imagen del conjunto de prueba
    #mostrar_imagen(X_test, 1)
    #exportar_a_onnx(model, (IMG_SIZE[0], IMG_SIZE[1], 3), "modelo_frutas.onnx")
    exportar_a_onnx(model, (128, 128, 3), "modelo_frutas.onnx")

    #validar_predicciones(model,
    #                 '/Users/cesartrincado/mango1.png',
    #                 '/Users/cesartrincado/platano4.png',
    #                 CLASSES,
    #                 IMG_SIZE)
    #predecir_imagen('/Users/cesartrincado/mango1.png', model)
