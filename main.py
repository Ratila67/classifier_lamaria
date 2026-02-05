import numpy
import os
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tqdm import tqdm

#Chargement des données
dataset_path = "malaria_hematie_dataset"
import numpy
import os
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models

#Chargement des données
dataset_path = "malaria_hematie_dataset"

#Creation de imagedatagenerator pour augmenter les données
train_datagen = ImageDataGenerator(
    rescale= 1./255,  # Normalisation
    rotation_range=20,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=True,
    validation_split=0.2  # Séparation automatique train/validation
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisation seulement, pas d'augmentation
    validation_split=0.2
)

# Chargement des données avec flow_from_directory
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # Taille plus raisonnable
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

#Creation du modele
mon_modele_malaria = models.Sequential(
    [
        keras.Input(shape=(224, 224, 3)),  # Changé de 1024 à 224
        keras.layers.Conv2D(100, kernel_size=(3,3), strides=(1,1), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(2,2)),

        keras.layers.Conv2D(100, kernel_size=(3,3), strides=(1,1), activation="relu"),
        keras.layers.MaxPool2D(pool_size=(2,2)),

        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ]
)

mon_modele_malaria.summary()

mon_modele_malaria.compile(loss='binary_crossentropy', optimizer='adam')

mon_modele_malaria.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

plt.figure(figsize=(10, 5))
plt.plot(mon_modele_malaria.history.history['loss'])
plt.title("Loss pendant l'entraînement")
plt.show()

mon_modele_malaria.save('mon_modele_malaria.h5')

#Test du modele
test_loss = mon_modele_malaria.evaluate(validation_generator)
print("Test loss:", test_loss)