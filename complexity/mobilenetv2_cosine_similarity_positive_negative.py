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
# import the timeit module
import timeit

# start the timer
start = timeit.default_timer()

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
layer_cosine_list_pos = []
layer_cosine_list_neg = []

layer_final_cos_list_pos = []
layer_final_cos_list_neg = []

for layer in model.layers[0:-2]:
    if 'conv' not in layer.name:
        continue

    index = getLayerIndex(model, layer.name)
    # print(layer.name)
    # append the convolved layer
    convolved_layers.append(index)
    convolved_layer_names.append(layer)

jdk_pos = {}
jdk_neg = {}

for lyr in convolved_layer_names:
    ary = np.array(lyr.get_weights(), dtype=object)
    layer_index = getLayerIndex(model, lyr.name)
    if len(ary) != 0:
        # filters, biases = layer.get_weights()
        # print(filters.shape)
        ary = np.array(lyr.get_weights(), dtype=object)

        zipper_dict_pos = {}
        zipper_dict_neg = {}
        dict_array = []

        # check for the arrays
        for x in ary:
            # print(len(x))
            # find if the array is 1 dim
            if x.ndim > 1:
                #for positives
                for y in x:
                    for z in y:
                        u = 0
                        for za in z:
                            # get all the positive elements in this and push them into an array
                            positive_elements = []
                            negative_elements = []
                            for pos_item in za:
                                if pos_item > 0:
                                    positive_elements.append(pos_item)
                            u = u + 1
                            zipper_dict_pos.update({u: positive_elements})

                #for negatives
                for y in x:
                    for z in y:
                        u = 0
                        for za in z:
                            # get all the negatives elements in this and push them into an array
                            negative_elements = []

                            for neg_item in za:
                                if neg_item < 0:
                                    negative_elements.append(neg_item)
                            u = u + 1
                            zipper_dict_neg.update({u: negative_elements})

            elif x.ndim == 1:
                # get the items/arrays with more than the 1 dimension
                print("")
        # print("lyr:",lyr.name)
        # get two items per positives
        for i in range(1, len(zipper_dict_pos), 2):
            if len(zipper_dict_pos[i]) > 0:
                if len(zipper_dict_pos[i]) == len(zipper_dict_pos[i + 1]):
                    new_i = np.array(zipper_dict_pos[i]).reshape(1, -1)
                    new_i_plus_1 = np.array(zipper_dict_pos[i + 1]).reshape(1, -1)
                    cos_item = np.ravel(cosine_similarity(new_i, new_i_plus_1))
                    layer_cosine_list_pos.append(cos_item)

                elif len(zipper_dict_pos[i]) > len(zipper_dict_pos[i + 1]):

                    diff_len = len(np.array(zipper_dict_pos[i])) - len(np.array(zipper_dict_pos[i + 1]))
                    old_i = np.pad(zipper_dict_pos[i + 1], (0, diff_len), 'constant')

                    new_i = old_i.reshape(1, -1)
                    new_i_plus_1 = np.array(zipper_dict_pos[i]).reshape(1, -1)

                    cos_item = np.ravel(cosine_similarity(new_i, new_i_plus_1))
                    layer_cosine_list_pos.append(cos_item)

                elif len(zipper_dict_pos[i]) < len(zipper_dict_pos[i + 1]):
                    # print("######less than i plus 1#########")

                    diff_len = len(zipper_dict_pos[i + 1]) - len(zipper_dict_pos[i])
                    old_i = np.pad(zipper_dict_pos[i], (0, diff_len), 'constant')
                    new_i = old_i.reshape(1, -1)
                    new_i_plus_1 = np.array(zipper_dict_pos[i + 1]).reshape(1, -1)

                    cos_item = np.ravel(cosine_similarity(new_i, new_i_plus_1))
                    layer_cosine_list_pos.append(cos_item)
        #for the negatives
        for i in range(1, len(zipper_dict_neg), 2):
            if len(zipper_dict_neg[i]) > 0:
                if len(zipper_dict_neg[i]) == len(zipper_dict_neg[i + 1]):
                    new_i = np.array(zipper_dict_neg[i]).reshape(1, -1)
                    new_i_plus_1 = np.array(zipper_dict_neg[i + 1]).reshape(1, -1)
                    cos_item = np.ravel(cosine_similarity(new_i, new_i_plus_1))
                    layer_cosine_list_neg.append(cos_item)

                elif len(zipper_dict_neg[i]) > len(zipper_dict_neg[i + 1]):

                    diff_len = len(np.array(zipper_dict_neg[i])) - len(np.array(zipper_dict_neg[i + 1]))
                    old_i = np.pad(zipper_dict_neg[i + 1], (0, diff_len), 'constant')

                    new_i = old_i.reshape(1, -1)
                    new_i_plus_1 = np.array(zipper_dict_neg[i]).reshape(1, -1)

                    cos_item = np.ravel(cosine_similarity(new_i, new_i_plus_1))
                    layer_cosine_list_neg.append(cos_item)

                elif len(zipper_dict_neg[i]) < len(zipper_dict_neg[i + 1]):
                    # print("######less than i plus 1#########")

                    diff_len = len(zipper_dict_neg[i + 1]) - len(zipper_dict_neg[i])
                    old_i = np.pad(zipper_dict_neg[i], (0, diff_len), 'constant')
                    new_i = old_i.reshape(1, -1)
                    new_i_plus_1 = np.array(zipper_dict_neg[i + 1]).reshape(1, -1)

                    cos_item = np.ravel(cosine_similarity(new_i, new_i_plus_1))
                    layer_cosine_list_neg.append(cos_item)


        # calculate softmax function
        def softmax(s_et):
            e_set = np.exp(s_et - np.max(s_et))
            return e_set / e_set.sum(axis=0)


        for a in layer_cosine_list_pos:
            layer_final_cos_list_pos.append(a)
        # print(layer_final_cos_list)
        layer_softmax_values = softmax(np.ravel(np.array(layer_cosine_list_pos, dtype=np.float32).reshape(1, -1)))
        jdk_pos.update({str(layer_index)+"-": layer_softmax_values})

        for a in layer_cosine_list_neg:
            layer_final_cos_list_neg.append(a)
        # print(layer_final_cos_list)
        layer_softmax_values_neg = softmax(np.ravel(np.array(layer_cosine_list_neg, dtype=np.float32).reshape(1, -1)))
        jdk_neg.update({str(layer_index)+"-": layer_softmax_values_neg})

b = [x for y, x in enumerate(jdk_pos) if y != i]
all_items = list(combinations(b, 2))

b1 = [x for y, x in enumerate(jdk_neg) if y != i]
all_items1 = list(combinations(b, 2))
# put the layers in a dictionary and rank them
other = []

#for positives
for a in all_items:
    if len(jdk_pos[a[0]]) > len(jdk_pos[a[1]]):
        dif_1 = len(np.array(jdk_pos[a[0]])) - len(np.array(jdk_pos[a[1]]))
        new_jdk = np.pad(np.array(jdk_pos[a[1]]), (0, dif_1), 'constant').reshape(1, -1)
        new_i_jdk = np.array(jdk_pos[a[0]]).reshape(1, -1)
        kl_pq = rel_entr(np.ravel(np.ravel(new_i_jdk), new_jdk))
        ######print("layer:", a[0], " and layer: ", a[1], ':KL(P ||2 Q): %.3f nats1' % sum(kl_pq))
        m_no = [a[0], a[1], sum(kl_pq)]
        other.append(m_no)

    elif len(jdk_pos[a[0]]) < len(jdk_pos[a[1]]):
        dif_1 = len(np.array(jdk_pos[a[1]])) - len(np.array(jdk_pos[a[0]]))
        new_jdk = np.pad(np.array(jdk_pos[a[0]]), (0, dif_1), 'constant').reshape(1, -1)
        new_i_jdk = np.array(jdk_pos[a[1]]).reshape(1, -1)
        kl_pq = rel_entr(np.ravel(new_jdk), np.ravel(new_i_jdk))
        ############print("layer:", a[0], " and layer: ", a[1], ':KL(P ||2 Q): %.3f nats' % sum(kl_pq))
        m_no = [a[0], a[1], sum(kl_pq)]
        other.append(m_no)

#for ngs
for a in all_items1:
    if len(jdk_neg[a[0]]) > len(jdk_neg[a[1]]):
        dif_1 = len(np.array(jdk_neg[a[0]])) - len(np.array(jdk_neg[a[1]]))
        new_jdk = np.pad(np.array(jdk_neg[a[1]]), (0, dif_1), 'constant').reshape(1, -1)
        new_i_jdk = np.array(jdk_neg[a[0]]).reshape(1, -1)
        kl_pq = rel_entr(np.ravel(np.ravel(new_i_jdk), new_jdk))
        ###############print("layer:", a[0], " and layer: ", a[1], ':KL(P ||2 Q): %.3f nats1' % sum(kl_pq))
        m_no = [a[0], a[1], sum(kl_pq)]
        other.append(m_no)

    elif len(jdk_neg[a[0]]) < len(jdk_neg[a[1]]):
        dif_1 = len(np.array(jdk_neg[a[1]])) - len(np.array(jdk_neg[a[0]]))
        new_jdk = np.pad(np.array(jdk_neg[a[0]]), (0, dif_1), 'constant').reshape(1, -1)
        new_i_jdk = np.array(jdk_neg[a[1]]).reshape(1, -1)
        kl_pq = rel_entr(np.ravel(new_jdk), np.ravel(new_i_jdk))
        ##############print("layer:", a[0], " and layer: ", a[1], ':KL(P ||2 Q): %.3f nats' % sum(kl_pq))
        m_no = [a[0], a[1], sum(kl_pq)]
        other.append(m_no)

# with open('mobilenet_kullback_positive_neg.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)  # write the header
#     writer.writerow(["Layer1", "Layer2", "DKL"])
#     for l_item in other:
#         writer.writerow(l_item)
    # All the program statements
stop = timeit.default_timer()
execution_time = stop - start
print("execution time:-", execution_time)