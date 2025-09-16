import tensorflow as tf
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim import Adam
#from sklearn.metrics import confusion_matrix, classification_report
#from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import itertools
import time
import os
import cv2
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,  TensorDataset, random_split
from PIL import Image



#Funkcja saveResize:
#1.Cropowanie maksymalnej liczby zdjęć o podanych wymiarach z każdego zdjęcia ze zbioru
#2.Zapisanie nowych zdjęć do nowego folderu
#3.Tworzenie folderu do zapisu zdjęć jeśli go nie ma

def saveResize(dataset, image, idx, height, width): #nazwa datasetu, zdjęcie, wysokość, szerokość
  pathH = str(height)
  pathW = str(width)
  path = f"Crops/cropped_{pathH}x{pathW}/"
  os.makedirs(path, exist_ok=True) #jeśli jeszcze nie ma takiego folderu to go tworzy

  maxH, maxW,  c = image.shape
  crop = 0
  h = 0
  while h+height<=maxH:
    w = 0
    while w+width<=maxW:
      croppedImg = image[h:(h+height), w:(w+width)]
      savePath = os.path.join(path, f"{dataset}_photo{idx}_crop{crop}.jpg")
      if not os.path.exists(savePath): #zapisywanie cropa tylko jeśli nie ma go w datasecie
        cv2.imwrite(savePath, croppedImg)
        crop+=1
      w+=width
    h+=height
  return None



#resizowanie zdjęć

h = 128 #wybrana wysokość
w = 128 #wybrana szerokość

dataset = "Datasets/Movie1/" #wybór datasetu
images = os.listdir(dataset)

for idx, i in enumerate(images): #iteracja po zdjęciach
  image = cv2.imread(dataset+i)
  saveResize(dataset.split("/")[1], image, idx, h, w) #nazwa datasetu, ścieżka do zapisu, zdjęcie, indeks zdjęcia, wysokość, szerokość
  #if idx == 1: #stop jeśli nie chcemy ciąć całego zbioru
    #print("Done")
    #break
  if idx%100==0:
    print(f"Status cięcia: {idx}")
print("Done")