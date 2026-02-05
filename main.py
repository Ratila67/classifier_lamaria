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

#Stockage des images et labels
images = []
labels = [] #0 pour uninfected, 1 pour parasitized

print("Chargement des images parasitées...")
parasitized_images = os.path.join(dataset_path, "parasitized")
for filename in tqdm(os.listdir(parasitized_images)): #tqdm pour afficher la progression, plus sympa
    img_path = os.path.join(parasitized_images, filename)  #chemin de l'image
    img = image.load_img(img_path, target_size=(1024, 1024)) #chargement de l'image
    img_array = image.img_to_array(img) #conversion de l'image en array
    images.append(img_array) #ajout de l'image à la liste des images initialisé avant
    labels.append(1) #1 pour parasitized

print("Chargement des images non parasitées...")
uninfected_images = os.path.join(dataset_path, "uninfected")
for filename in tqdm(os.listdir(uninfected_images)):
    img_path = os.path.join(uninfected_images, filename)
    img = image.load_img(img_path, target_size=(1024, 1024))
    img_array = image.img_to_array(img)
    images.append(img_array)
    labels.append(0) #0 pour uninfected

print("Chargement des images terminé")

#convertir en numpy array
images = numpy.array(images)
labels = numpy.array(labels)

#afficher les dimensions des images et labels
print("Dimensions des images:", images.shape)
print("Dimensions des labels:", labels.shape)

print("Nombre d'images total chargées:", len(images))

images = images / 255 #normalisation des images

#Creation de imagedatagenerator pour augmenter les données
train_datagen = ImageDataGenerator(
    rotation_range=20, #rotation des images
    zoom_range=0.2, #zoom des images
    vertical_flip=True, #retournement vertical des images
    horizontal_flip=True, #retournement horizontal des images
)

test_datagen = ImageDataGenerator() #on ne fait pas d'augmentation des données pour le test

#Ici on sépare le jeu de données en train et test
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
test_generator = test_datagen.flow(x_test, y_test, batch_size=32)

sample_image = x_train[0]
plt.imshow(sample_image)
plt.show()

sample_label = y_train[0]
print("Label de l'image sample:", sample_label)
#Creation du modele
mon_modele_malaria = models.Sequential(
    [
        keras.input(shape=(1024, 1024, 3)),
        keras.layers.Conv2D(100, kernel_size=(3,3), strides=(1,1), activation="relu"),#convolution et activation
        keras.layers.MaxPool2D(pool_size=(2,2)), #pooling

        keras.layers.Conv2D(100, kernel_size=(3,3), strides=(1,1), activation="relu"),#convolution et activation
        keras.layers.MaxPool2D(pool_size=(2,2)), #pooling

        #Ensuite on fait la connection avec un vecteur flatten

        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ]
)

mon_modele_malaria.summary()

mon_modele_malaria.compile(loss='binary_crossentropy', optimizer='adam')

mon_modele_malaria.fit(train_generator,
                        steps_per_epoch=len(x_train) // 32, #Nombre de batches par epoch (en faisant // on arrondit au nombre entier inférieur, sinon bug)
                        epochs=10,
                        validation_data=test_generator, # données de validation
                        validation_steps=len(x_test) // 32) #Nombre de batches par epoch de validation (en faisant // on arrondit au nombre entier inférieur)


plt.figure(figsize=(10, 5))
plt.plot(mon_modele_malaria.history.history['loss'])
plt.title("Loss pendant l'entraînement")
plt.show()

mon_modele_malaria.save('mon_modele_malaria.h5')

#Test du modele
test_loss = mon_modele_malaria.evaluate(x_test, y_test)
print("Test loss:", test_loss)

#commentaire de test pour le commit