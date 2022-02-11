import numpy as np
from numpy.linalg import norm
import sklearn
from tensorflow import keras
import string
from scipy.special import rel_entr
from tensorflow.keras.applications.resnet50 import ResNet50 as resnet50
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.applications.densenet import DenseNet169 as densenet
from tensorflow.keras.applications.inception_v3 import InceptionV3 as inceptionv3
from sklearn.metrics.pairwise import cosine_similarity
import csv

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
layer_final_cos_list = []

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

jdk = {}
layered_jdk = {}
e = 0
pos_ves = []
neg_ves = []

# get the name of the layers

layer_names = []


# get the indexes of the layers
# get the layer index
def getLayerIndex(model_i, layer_name):
    for pos, layer_g in enumerate(model_i.layers):
        if layer_g.name == layer_name:
            return pos


for lyr in convolved_layer_names:
    print(str(getLayerIndex(model, lyr.name)), lyr.name)
    layer_names.append(str(getLayerIndex(model, lyr.name)) + lyr.name)
    e = e + 1
    # print("############", lyr.name, "###################")
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
                            negative_elements = []
                            for pos_item in za:
                                # for the positives
                                if pos_item < 0:
                                    positive_elements.append(pos_item)
                                    pos_ves.append(pos_item)
                                elif pos_item > 0:
                                    negative_elements.append(pos_item)
                            u = u + 1
                            zipper_dict.update({u: negative_elements})
                            # print("layer ", lyr.name," positives: ", len(positive_elements)," negatives: ", len(negative_elements))
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


        # print("layer ", lyr.name, " positives: ", len(pos_ves), " negatives: ", len(neg_ves))

        # calculate softmax function
        def softMax(s_et):
            e_set = np.exp(s_et - np.max(s_et))
            return e_set / e_set.sum(axis=0)


        for a in layer_cosine_list:
            layer_final_cos_list.append(a)
            # print(layer_final_cos_list)
        layer_softmax_values = softMax(np.ravel(np.array(layer_cosine_list, dtype=np.float32).reshape(1, -1)))

        jdk.update({e: layer_softmax_values})
        layered_jdk.update({e: str(getLayerIndex(model,lyr.name))})

# print(jdk)

# then pick two sets each and check their kullback divergence
for x in range(1,len(layered_jdk)):
    print(x, layered_jdk[x])

s = 0
# put the layers in a dictionary and rank them
other = []
for i in range(len(jdk)):
    b = [x for y, x in enumerate(jdk) if y != i]
    s = 0

    for p in b:

        if len(jdk[p]) < len(jdk[i + 1]):
            # print("less than")
            # print("layer:", p, " and layer: ", i + 1)
            dif_1 = len(np.array(jdk[i + 1])) - len(np.array(jdk[p]))
            new_jdk = np.pad(jdk[p], (0, dif_1), 'constant').reshape(1, -1)
            new_i_jdk = np.array(jdk[i + 1]).reshape(1, -1)

            kl_pq = rel_entr(np.ravel(new_jdk), np.ravel(new_i_jdk))
            print("layer:", p, " and layer: ", i + 1, ':KL(P ||2 Q): %.3f nats' % sum(kl_pq))

            m_no = [p, i + 1, sum(kl_pq)]
            other.append(m_no)

with open('kullback_all.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)  # write the header
    writer.writerow(["Layer1", "Layer2", "DKL"])

    for l_item in other:
        writer.writerow(l_item)
# for t in layer_names:
#     print(t)
