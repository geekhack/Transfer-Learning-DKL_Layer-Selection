import numpy as np
from numpy.linalg import norm
import sklearn
from tensorflow import keras
import string
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.applications.densenet import DenseNet169 as densenet
from tensorflow.keras.applications.inception_v3 import InceptionV3 as inceptionv3
from tensorflow.keras.applications.efficientnet import EfficientNetB1 as efficient
from sklearn.metrics.pairwise import cosine_similarity

model_name = resnet50
input_t = keras.Input(shape=(224, 224, 3))
model = model_name(include_top=False,
                   weights="imagenet",
                   input_tensor=input_t)


# get the layer index
def getLayerIndex(model_i, layer_name):
    for pos, layer_g in enumerate(model_i.layers):
        if layer_g.name == layer_name:
            return pos


# get the convolved layers into an array for looping
convolved_layers = []
convolved_layer_names = []
# create a list to place them
layer_cosine_list = []

for layer in model.layers[0:-2]:

    t = np.array(layer.get_weights(), dtype=object).ndim
    ary = np.array(layer.get_weights(), dtype=object)

    if (model_name != resnet50) or (model_name != vgg16):
        if (len(ary) > 0) and (t > 2):
            index = getLayerIndex(model, layer.name)
            # print(layer.name)
            # append the convolved layer
            convolved_layers.append(index)
            convolved_layer_names.append(layer)
            # print(str(len(array)) + "for:" + layer.name + "at index:" + str(index))
    if (model_name == resnet50) or (model_name == vgg16):
        if len(ary) > 0 and (t != 2):
            index = getLayerIndex(model, layer)
            # print(layer.name)
            # append the convolved layer
            convolved_layers.append(index)
            convolved_layer_names.append(layer)


for lyr in convolved_layer_names:
    #print("############", lyr.name, "###################")
    ary = np.array(lyr.get_weights(), dtype=object)
    if len(ary) != 0:
        # filters, biases = layer.get_weights()
        # print(filters.shape)
        ary = np.array(lyr.get_weights(), dtype=object)

        zipper_dict = {}
        dict_array = []

        # check for the arrays
        for x in ary:
            # print(len(x))
            # find if the array is 1 dim
            if x.ndim > 1:
                for y in x:
                    for z in y:
                        u = 0
                        for za in z:
                            # get all the negative elements in this and push them into an array
                            positive_elements = []
                            for pos_item in za:
                                positive_elements.append(pos_item)
                                u = u + 1

                                zipper_dict.update({u: positive_elements})

            elif x.ndim == 1:
                # get the items/arrays with more than the 1 dimension
                print("")

        # get two items per tuple
        for i in range(1, len(zipper_dict), 2):

            if len(zipper_dict[i]) == len(zipper_dict[i + 1]):

                new_i = np.array(zipper_dict[i]).reshape(1, -1)
                new_i_plus_1 = np.array(zipper_dict[i + 1]).reshape(1, -1)
                cos_item = np.ravel(cosine_similarity(new_i, new_i_plus_1))
                layer_cosine_list.append(cos_item)

            elif len(zipper_dict[i]) > len(zipper_dict[i + 1]):

                diff_len = len(np.array(zipper_dict[i])) - len(np.array(zipper_dict[i + 1]))
                old_i = np.pad(zipper_dict[i + 1], (0, diff_len), 'constant')

                new_i = old_i.reshape(1, -1)
                new_i_plus_1 = np.array(zipper_dict[i]).reshape(1, -1)

                cos_item = np.ravel(cosine_similarity(new_i, new_i_plus_1))
                layer_cosine_list.append(cos_item)

            elif len(zipper_dict[i]) < len(zipper_dict[i + 1]):
                # print("######less than i plus 1#########")

                diff_len = len(zipper_dict[i + 1]) - len(zipper_dict[i])
                old_i = np.pad(zipper_dict[i], (0, diff_len), 'constant')
                new_i = old_i.reshape(1, -1)
                new_i_plus_1 = np.array(zipper_dict[i + 1]).reshape(1, -1)

                cos_item = np.ravel(cosine_similarity(new_i, new_i_plus_1))
                layer_cosine_list.append(cos_item)

    print(lyr.name, len(layer_cosine_list))

    #calculate softmax function
    def softmax(set):
        e_set = np.exp(set - np.max(set))
        return e_set / e_set.sum(axis=0)