import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.inception_v3 import InceptionV3 as inception
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.applications.densenet import DenseNet169 as densenet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

print(tf.__version__)
df = pd.read_csv("mobilenetv2_pure_kullback_positive.csv")
df['layer2_layer1'] = df['Layer2'] -df['Layer1']
lower_DKL = df.sort_values('DKL')
# # return the lower layers
ascending = lower_DKL.head()
print(ascending)
#
higher_DKL = df.sort_values('DKL', ascending=False)
# # return the lower layers
descending = higher_DKL.head()

# #create an array to store lower layers
new_dict =[]
for m in range(len(ascending)):
    new_dict.append(int(ascending.iloc[m]['Layer1']))
    new_dict.append(int(ascending.iloc[m]['Layer2']))

lower_DKL_layers = []
[lower_DKL_layers.append(x) for x in new_dict if x not in lower_DKL_layers]
lower_DKL_layers.sort()
print("lower layers",lower_DKL_layers)

#create an array to store upper kullback layers
upper_new_dict =[]
for n in range(len(descending)):
    upper_new_dict.append(int(descending.iloc[n]['Layer1']))
    upper_new_dict.append(int(descending.iloc[n]['Layer2']))

upper_DKL_layers = []
[upper_DKL_layers.append(z) for z in upper_new_dict if z not in upper_DKL_layers]
upper_DKL_layers.sort()
print("upper layers",upper_DKL_layers)