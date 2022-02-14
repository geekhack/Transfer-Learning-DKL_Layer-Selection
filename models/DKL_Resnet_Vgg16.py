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
import kullback_pos_neg_selection as pndkl
import kullback_pos_selection as pdkl
import kullback_neg_selection as ndkl

pndkl.getcsv()
pdkl.pgetcsv()
ndkl.n_getcsv()

# *******************positive negative kullback divergence************************************
df = pd.read_csv("../kullback_positive_negative.csv")
df['layer2_layer1'] = df['Layer2'] - df['Layer1']
# print(df)
updated_df = df.loc[df['layer2_layer1'] == 1]

lower_DKL = updated_df.sort_values('DKL')
# return the lower layers
ascending = lower_DKL.head()

higher_DKL = updated_df.sort_values('DKL', ascending=False)
# return the lower layers
descending = higher_DKL.head()
# create an array to store lower layers
new_dict = []
for m in range(len(ascending)):
    new_dict.append(int(ascending.iloc[m]['Layer1']))
    new_dict.append(int(ascending.iloc[m]['Layer2']))

lower_DKL_layers = []
[lower_DKL_layers.append(x) for x in new_dict if x not in lower_DKL_layers]

# sort list,the first 5
#print(lower_DKL_layers[:5])

# create an array to store upper kullback layers
upper_new_dict = []
for n in range(len(descending)):
    upper_new_dict.append(int(descending.iloc[n]['Layer1']))
    upper_new_dict.append(int(descending.iloc[n]['Layer2']))

upper_DKL_layers = []
[upper_DKL_layers.append(z) for z in upper_new_dict if z not in upper_DKL_layers]
##can use the upper_DKL_layers or the lower_DKL_layers depending on the need
# *************************positive DKL*******************************************
df_positive = pd.read_csv("../kullback_positive.csv")
df_positive['layer2_layer1'] = df_positive['Layer2'] - df_positive['Layer1']
# print(df)
positive_updated_df = df_positive.loc[df_positive['layer2_layer1'] == 1]

p_lower_DKL = positive_updated_df.sort_values('DKL')
# return the lower layers
p_ascending = p_lower_DKL.head()

p_higher_DKL = positive_updated_df.sort_values('DKL', ascending=False)
# return the lower layers
p_descending = p_higher_DKL.head()
# create an array to store lower layers
p_new_dict = []
for m in range(len(p_descending)):
    p_new_dict.append(int(p_ascending.iloc[m]['Layer1']))
    p_new_dict.append(int(p_ascending.iloc[m]['Layer2']))

p_lower_DKL_layers = []
[p_lower_DKL_layers.append(x) for x in p_new_dict if x not in p_lower_DKL_layers]

# sort list,the first 5
#print(p_lower_DKL_layers[:5])

# create an array to store upper kullback layers
p_upper_new_dict = []
for n in range(len(p_descending)):
    p_upper_new_dict.append(int(p_descending.iloc[n]['Layer1']))
    p_upper_new_dict.append(int(p_descending.iloc[n]['Layer2']))

p_upper_DKL_layers = []
[p_upper_DKL_layers.append(z) for z in p_upper_new_dict if z not in p_upper_DKL_layers]
# ***********************negative DKL*********************************************
df_negative = pd.read_csv("../kullback_negative.csv")
df_negative['layer2_layer1'] = df_negative['Layer2'] - df_negative['Layer1']
# print(df)
n_updated_df = df_negative.loc[df_negative['layer2_layer1'] == 1]

n_lower_DKL = n_updated_df.sort_values('DKL')
# return the lower layers
n_ascending = n_lower_DKL.head()

n_higher_DKL = n_updated_df.sort_values('DKL', ascending=False)
# return the lower layers
n_descending = n_higher_DKL.head()
# create an array to store lower layers
n_new_dict = []
for m in range(len(n_ascending)):
    n_new_dict.append(int(n_ascending.iloc[m]['Layer1']))
    n_new_dict.append(int(n_ascending.iloc[m]['Layer2']))

n_lower_DKL_layers = []
[n_lower_DKL_layers.append(x) for x in n_new_dict if x not in n_lower_DKL_layers]

# sort list,the first 5
#print(n_lower_DKL_layers[:5])

# create an array to store upper kullback layers
n_upper_new_dict = []
for n in range(len(n_descending)):
    n_upper_new_dict.append(int(n_descending.iloc[n]['Layer1']))
    n_upper_new_dict.append(int(n_descending.iloc[n]['Layer2']))

n_upper_DKL_layers = []
[n_upper_DKL_layers.append(z) for z in n_upper_new_dict if z not in n_upper_DKL_layers]


# can pick the lower/higher layers depending on the kullback to check
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

input_shape = (32, 32, 3)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_train = x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_test = x_test / 255.0

y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

inputs = tf.keras.layers.Input(shape=(32, 32, 3))
model_name = resnet50
input_t = (32,32,3)
model = model_name(include_top=False,
                   weights="imagenet",
                   input_shape=input_t)
model2 = model_name(include_top=False,
                   weights="imagenet",
                   input_shape=input_t)

model3 = model_name(include_top=False,
                   weights="imagenet",
                   input_shape=input_t)

model_tune1 = model_name(include_top=False,
                   weights="imagenet",
                   input_shape=input_t)
model_tune2 = model_name(include_top=False,
                   weights="imagenet",
                   input_shape=input_t)
model_tune3 = model_name(include_top=False,
                   weights="imagenet",
                   input_shape=input_t)
model_tune4 = model_name(include_top=False,
                   weights="imagenet",
                   input_shape=input_t)


# get the layer index
def getLayerIndex(model_i, layer_name):
    for pos, layer_g in enumerate(model_i.layers):
        if layer_g.name == layer_name:
            return pos


# for DKL layers
pcs_h_layers = p_lower_DKL_layers   # positive DKL,,
pcs_l_layers = n_lower_DKL_layers  # negative DKL
pncs_l_layers = lower_DKL_layers  # positive negative DKL
for sb_layer in model.layers:
    # sb_layer.trainable = False
    index = getLayerIndex(model, sb_layer.name)
    # for b in final_selected_layers:
    for b in pcs_h_layers:
        if b == index:
            sb_layer.trainable = True
            # print(str(sb_layer.name) + " and index is" + str(b))
            print(sb_layer.name, sb_layer.trainable)

for sbs_layer in model2.layers:
    # sb_layer.trainable = False
    index = getLayerIndex(model2, sbs_layer.name)
    # for b in final_selected_layers:
    for b in pcs_l_layers:
        if b == index:
            sb_layer.trainable = True
            # print(str(sb_layer.name) + " and index is" + str(b))
            print(sbs_layer.name, sbs_layer.trainable)

for sby_layer in model3.layers:
    # sb_layer.trainable = False
    index = getLayerIndex(model3, sby_layer.name)
    # for b in final_selected_layers:
    for b in pncs_l_layers:
        if b == index:
            sby_layer.trainable = True
            # print(str(sb_layer.name) + " and index is" + str(b))
            print(sby_layer.name, sby_layer.trainable)

# finetune by removeing the last layer
for lst_layer in model_tune1.layers[:-2]:
    lst_layer.trainable = False

####end of the last layer
# finetune by removeing the 2nd last layer
for scnd_st_layer in model_tune2.layers[:-3]:
    scnd_st_layer.trainable = False

####end of the last layer
# finetune by removeing the 3rd last layer
for thrd_layer in model_tune3.layers[:-4]:
    thrd_layer.trainable = False

####end of the last layer
# for feature extraction
for ftr_layer in model_tune4.layers:
    ftr_layer.trainable = False



x = layers.Flatten()(model.output)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(10, activation = 'softmax')(x)
t_model = Model(inputs = model.input, outputs = predictions)

a = layers.Flatten()(model2.output)
a = layers.Dense(4096, activation='relu')(a)
a = layers.Dropout(0.5)(a)
a = layers.Dense(4096, activation='relu')(a)
a = layers.Dropout(0.5)(a)
predictions = layers.Dense(10, activation = 'softmax')(a)
t_model2 = Model(inputs = model2.input, outputs = predictions)

b = layers.Flatten()(model3.output)
b = layers.Dense(4096, activation='relu')(b)
b = layers.Dropout(0.5)(b)
b = layers.Dense(4096, activation='relu')(b)
b = layers.Dropout(0.5)(b)
predictionsb = layers.Dense(10, activation = 'softmax')(b)
t_model3 = Model(inputs = model3.input, outputs = predictionsb)

c = layers.Flatten()(model_tune1.output)
c = layers.Dense(4096, activation='relu')(c)
c = layers.Dropout(0.5)(c)
c = layers.Dense(4096, activation='relu')(c)
c = layers.Dropout(0.5)(c)
predictions = layers.Dense(10, activation = 'softmax')(c)
t_model4 = Model(inputs = model_tune1.input, outputs = predictions)

d = layers.Flatten()(model_tune2.output)
d = layers.Dense(4096, activation='relu')(d)
d = layers.Dropout(0.5)(d)
d = layers.Dense(4096, activation='relu')(d)
d = layers.Dropout(0.5)(d)
predictions = layers.Dense(10, activation = 'softmax')(d)
t_model5 = Model(inputs = model_tune2.input, outputs = predictions)

e = layers.Flatten()(model_tune3.output)
e = layers.Dense(4096, activation='relu')(e)
e = layers.Dropout(0.5)(e)
e = layers.Dense(4096, activation='relu')(e)
e = layers.Dropout(0.5)(e)
predictions = layers.Dense(10, activation = 'softmax')(e)
t_model6 = Model(inputs = model_tune3.input, outputs = predictions)

f = layers.Flatten()(model_tune4.output)
f = layers.Dense(4096, activation='relu')(f)
f = layers.Dropout(0.5)(f)
f = layers.Dense(4096, activation='relu')(f)
f = layers.Dropout(0.5)(f)
predictions = layers.Dense(10, activation = 'softmax')(f)
t_model7 = Model(inputs = model_tune4.input, outputs = predictions)



t_model.compile(optimizer=optimizers.SGD(lr=1e-5,momentum=0.9), loss=losses.categorical_crossentropy, metrics=['accuracy'])
history = t_model.fit(x_train, y_train, batch_size=64, shuffle=True,validation_data=(x_test, y_test), epochs=50, verbose=1)

#for model2
t_model2.compile(optimizer=optimizers.SGD(lr=1e-5,momentum=0.9), loss=losses.categorical_crossentropy, metrics=['accuracy'])
history2 = t_model2.fit(x_train, y_train, batch_size=64, shuffle=True,validation_data=(x_test, y_test), epochs=50, verbose=1)

#for model2
t_model3.compile(optimizer=optimizers.SGD(lr=1e-5,momentum=0.9), loss=losses.categorical_crossentropy, metrics=['accuracy'])
history3 = t_model3.fit(x_train, y_train, batch_size=64, shuffle=True,validation_data=(x_test, y_test), epochs=50, verbose=1)

#for model3
t_model4.compile(optimizer=optimizers.SGD(lr=1e-5,momentum=0.9), loss=losses.categorical_crossentropy, metrics=['accuracy'])
history4= t_model4.fit(x_train, y_train, batch_size=64, shuffle=True,validation_data=(x_test, y_test), epochs=50, verbose=1)


#for model3
t_model5.compile(optimizer=optimizers.SGD(lr=1e-5,momentum=0.9), loss=losses.categorical_crossentropy, metrics=['accuracy'])
history5 = t_model5.fit(x_train, y_train, batch_size=64, shuffle=True,validation_data=(x_test, y_test), epochs=50, verbose=1)


#for model3
t_model6.compile(optimizer=optimizers.SGD(lr=1e-5,momentum=0.9), loss=losses.categorical_crossentropy, metrics=['accuracy'])
history6 = t_model6.fit(x_train, y_train, batch_size=64, shuffle=True,validation_data=(x_test, y_test), epochs=50, verbose=1)

#for model3
t_model7.compile(optimizer=optimizers.SGD(lr=1e-5,momentum=0.9), loss=losses.categorical_crossentropy, metrics=['accuracy'])
history7 = t_model7.fit(x_train, y_train, batch_size=64, shuffle=True,validation_data=(x_test, y_test), epochs=50, verbose=1)



plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.plot(history2.history['loss'])
#plt.plot(history2.history['val_loss'])
plt.plot(history3.history['loss'])
plt.plot(history4.history['loss'])
#plt.plot(history4.history['val_loss'])
plt.plot(history5.history['loss'])
#plt.plot(history5.history['val_loss'])
plt.plot(history6.history['loss'])
#plt.plot(history6.history['val_loss'])
plt.plot(history7.history['loss'])
#plt.plot(history6.history['val_loss'])
plt.title('')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['PDKL','NDKL','PNDKL','1st_Layer','2nd_Layer','3rd_Layer','Feature_Extraction'], loc="center right", bbox_to_anchor=(1.5, 0.5))
#plt.legend(['PCS_train', 'PCS_val','NCS_train','NCS_val','PNCS_train','PNCS_val'], loc='upper right')
plt.savefig('resnet_cifar10_loss.eps', dpi=1000, format="eps", bbox_inches="tight")
plt.show()
