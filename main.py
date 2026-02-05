import numpy
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
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

#afficher les premieres images et labels
print("Premieres images:", images[0])
print("Premieres labels:", labels[0])
