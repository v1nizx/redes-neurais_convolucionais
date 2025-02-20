# Bibliotecas necessárias
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from PIL import Image
from google.colab import drive
from sklearn.metrics import classification_report

# Bibliotecas do Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Montar o Google Drive
drive.mount('/content/drive')

# Definições iniciais
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DATASET_PATH = '/content/drive/MyDrive/Disciplina Sistemas Inteligentes/Bases/Railway Track fault Detection Updated'  # Atualize para o caminho correto

# Configurar geradores de imagens
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'Train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'Train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATASET_PATH, 'Test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Construir a CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_generator)
print(f"Acurácia no teste: {test_acc:.4f}")

# Avaliação detalhada
true_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)
print(classification_report(true_labels, predicted_labels, target_names=test_generator.class_indices.keys()))

# Configurar gerador de validação
val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATASET_PATH, 'Validation'),  # Certifique-se de que existe uma pasta 'validation'
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Treinamento com validação
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Avaliação no conjunto de validação
val_loss, val_acc = model.evaluate(val_generator)
print(f"Acurácia na validação: {val_acc:.4f}")

# Relatório de classificação na validação
true_labels_val = val_generator.classes
predictions_val = model.predict(val_generator)
predicted_labels_val = np.argmax(predictions_val, axis=1)
print(classification_report(true_labels_val, predicted_labels_val, target_names=val_generator.class_indices.keys()))

