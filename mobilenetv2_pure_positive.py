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
from tensorflow.keras.applications.efficientnet import EfficientNetB1 as efficient
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as mobilenetv2
from tensorflow.keras.applications.mobilenet import MobileNet as mobilenet
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import csv

model_name = mobilenetv2
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
    index = getLayerIndex(model, layer.name)

    if ('depthwise' in layer.name) and ('depthwise_BN' not in layer.name) and ('depthwise_relu' not in layer.name):
        convolved_layers.append(index)
        convolved_layer_names.append(layer)
    if ('project' in layer.name) and ('project_BN' not in layer.name) :
        convolved_layers.append(index)
        convolved_layer_names.append(layer)
    if ('conv' in layer.name) and ('conv_depthwise_BN' not in layer.name) and ('conv_project_BN' not in layer.name) and ('conv_depthwise_relu' not in layer.name):
        convolved_layers.append(index)
        convolved_layer_names.append(layer)
        #print(layer.name)or ('depthwise_relu' not in layer.name))

    #print(layer.name)
    # append the convolved layer


jdk = {}
for lyr in convolved_layer_names:
    ary = np.array(lyr.get_weights(), dtype=object)
    layer_index = getLayerIndex(model, lyr.name)
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
                            # get all the positive elements in this and push them into an array
                            positive_elements = []
                            for pos_item in za:
                                if pos_item > 0:
                                    positive_elements.append(pos_item)
                            u = u + 1
                            zipper_dict.update({u: positive_elements})

            elif x.ndim == 1:
                # get the items/arrays with more than the 1 dimension
                print("")
        # print("lyr:",lyr.name)
        # get two items per tuple
        for i in range(1, len(zipper_dict), 2):
            if len(zipper_dict[i]) > 0:
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


        # calculate softmax function
        def softmax(s_et):
            e_set = np.exp(s_et - np.max(s_et))
            return e_set / e_set.sum(axis=0)


        for a in layer_cosine_list:
            layer_final_cos_list.append(a)
        # print(layer_final_cos_list)
        layer_softmax_values = softmax(np.ravel(np.array(layer_cosine_list, dtype=np.float32).reshape(1, -1)))
        jdk.update({layer_index: layer_softmax_values})

# jdk gives a dictionary made up of probability distributions
# print(jdk)
# print("the other one is")
b = [x for y, x in enumerate(jdk) if y != i]
all_items = list(combinations(b, 2))
# put the layers in a dictionary and rank them
other = []

for a in all_items:
    if len(jdk[a[0]]) > len(jdk[a[1]]):
        dif_1 = len(np.array(jdk[a[0]])) - len(np.array(jdk[a[1]]))
        new_jdk = np.pad(np.array(jdk[a[1]]), (0, dif_1), 'constant').reshape(1, -1)
        new_i_jdk = np.array(jdk[a[0]]).reshape(1, -1)
        kl_pq = rel_entr(np.ravel(np.ravel(new_i_jdk), new_jdk))
        print("layer:", a[0], " and layer: ", a[1], ':KL(P ||2 Q): %.3f nats1' % sum(kl_pq))
        m_no = [a[0], a[1], sum(kl_pq)]
        other.append(m_no)

    elif len(jdk[a[0]]) < len(jdk[a[1]]):
        dif_1 = len(np.array(jdk[a[1]])) - len(np.array(jdk[a[0]]))
        new_jdk = np.pad(np.array(jdk[a[0]]), (0, dif_1), 'constant').reshape(1, -1)
        new_i_jdk = np.array(jdk[a[1]]).reshape(1, -1)
        kl_pq = rel_entr(np.ravel(new_jdk), np.ravel(new_i_jdk))
        print("layer:", a[0], " and layer: ", a[1], ':KL(P ||2 Q): %.3f nats' % sum(kl_pq))
        m_no = [a[0], a[1], sum(kl_pq)]
        other.append(m_no)

with open('mobilenetv2_pure_kullback_positive.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)  # write the header
    writer.writerow(["Layer1", "Layer2", "DKL"])
    for l_item in other:
        writer.writerow(l_item)
