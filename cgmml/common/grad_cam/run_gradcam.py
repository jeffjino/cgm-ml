# gradcam algorithm taken from F. Chollet elephant example
import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

REPO_DIR = Path(os.getcwd()).parents[2].absolute()

# from cgm-rg
IMAGE_TARGET_HEIGHT = 240
IMAGE_TARGET_WIDTH = 180
NORMALIZATION_VALUE = 7.5


def show_depthmaps(depthmaps):
    for depthmap in depthmaps:
        plt.imshow(depthmap, cmap='gray')
        plt.show()


def show_heatmaps(heatmaps):
    for heatmap in heatmaps:
        plt.matshow(heatmap)
        plt.show()


def remove_batchdim(depthmaps):
    new_depthmaps = []
    for depthmap in depthmaps:
        depthmap = depthmap[0, :, :, :]
        new_depthmaps.append(depthmap)

    return new_depthmaps


def add_batchdim(depthmaps):
    new_depthmaps = []
    for depthmap in depthmaps:
        depthmap = np.expand_dims(depthmap, axis=0)
        new_depthmaps.append(depthmap)

    return new_depthmaps


def show_gradcam(superimposed_imgs):
    for img in superimposed_imgs:
        plt.matshow(img)
        plt.show()


def overlay_depthmap_gradcam(depthmap, gradcam, transparency=0.4):
    depthmap = depthmap.reshape(depthmap.shape[1:])
    depthmap_saved = keras.preprocessing.image.array_to_img(depthmap)
    depthmap_saved.save('depthmap.png')

    heatmap = np.uint8(255 * gradcam)

    jet_cm = cm.get_cmap("jet")
    jet_colors = jet_cm(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_hm_img = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_hm_img = jet_hm_img.resize((depthmap.shape[1], depthmap.shape[0]))
    jet_hm_img.save('jetmap.png')

    depthmap_load = cv2.imread('depthmap.png')
    depthmap_load = depthmap_load[..., ::-1]

    jet_load = cv2.imread('jetmap.png')
    jet_load = jet_load[..., ::-1]
    output_image = cv2.addWeighted(depthmap_load, (1.0 - transparency), jet_load, transparency, 0)
    #cv2.imwrite('jasmintest.png', output_image)

    plt.imshow(output_image)
    plt.show()
    return output_image


def extract_last_conv_layer_name(model, substring='conv'):
    # 1. save all layer names in a list
    layer_names = [layer.name for layer in model.layers]

    # 2. search for substring in all layers and put them in list
    conv_layer_names = []
    for layer in layer_names:
        if substring in layer:
            conv_layer_names.append(layer)

    # 3. return last one
    return conv_layer_names[-1]


# GRADCAM checks the importance of output filters (here: last conv layer) towards the final decision
def compute_preds_and_heatmaps(numpy_arrays, grad_model):
    heatmaps = []
    height_predictions = []

    for numpy_array in numpy_arrays:
        # GET the score for target prediction
        print(numpy_array.shape)
        numpy_array = np.expand_dims(numpy_array, axis=0) # IT DOES NOT WORK WITHOUT THIS - I CANNOT CHANGE HOW grad_model WORK INTERNALLY
        print(numpy_array.shape)
        with tf.GradientTape() as tape:
            numpy_array = tf.cast(numpy_array, tf.float32)
            (conv_outputs, height_prediction) = grad_model(np.array(numpy_array))
            print("pred.shape = ", height_prediction.shape)
            print("predictions", height_prediction)
            print("predictions[0]", height_prediction[0])
            # THIS is the difference between CLASS activation and regression: in class activation, we input the specific class index here, 
            # e.g. 2= cat. then, we read out the probability for cat in the flattened predictions at index "cat" and use this to get the activation from gradients
            # IN OUR CASE: we do not have classes, we do regression. So: we have one height value in the zeroth index of our predictions tensor. And we want to see where the 
            # activation was in the last conv layer for this prediction

            # IMPORTANT: Since we always will only have one prediction (1 depthmap is processed at a time, in regression we do not have multiple class predictions but only one scalar), 
            # we can leave the next block out
            #classIdx = np.argmax(height_predictions[0]) # returns index of prediction
            #loss = height_predictions[:, classIdx] 
        
        # EXTRACT filters and gradients
        output = conv_outputs[0]
        grads = tape.gradient(height_prediction, conv_outputs)[0]

        # GUIDED backpropagation - eliminating elements that act negatively towards the decision - zeroing-out negative gradients
        guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
        
        # AVERAGE gradients spatially
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))

        # BUILD a weighted map of filters according to gradients importance
        cam = np.ones(output.shape[0: 2], dtype = np.float32)
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # HEATMAP visualization
        cam = cv2.resize(cam.numpy(), (IMAGE_TARGET_HEIGHT, IMAGE_TARGET_WIDTH))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())
        
        heatmaps.append(heatmap)
        height_predictions.append(height_prediction)

    return heatmaps, height_predictions
