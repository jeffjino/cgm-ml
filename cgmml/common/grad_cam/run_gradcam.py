# gradcam algorithm taken from F. Chollet elephant example

import azureml
from azureml.core import Workspace, Dataset
from azureml.core import Experiment
from azureml.core.run import Run 
import sys
import os

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
#from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
import numpy as np
from pathlib import Path
import glob2 as glob
import random
import pickle
import cv2

# Display F. Chollet example
from IPython.display import Image, display
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage.transform import rescale, resize, downscale_local_mean

from cgmml.common.evaluation.eval_utilities import download_model

REPO_DIR = Path(os.getcwd()).parents[2].absolute()


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            print("pred_index = ", pred_index)
            print("preds[0] = ", preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(dmap_array, heatmap, cam_path, alpha=0.4):
    
    #dmap_array = dmap.depthmap_arr.reshape(1, 240, 180, 1)
    print(dmap_array.shape)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    print("jet_heatmap.shape", jet_heatmap.size)
    
    jet_heatmap = jet_heatmap.resize((dmap_array.shape[2], dmap_array.shape[1]))
    
    print("jet_heatmap.shape", jet_heatmap.size)
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    ## TODO - fix dimensions!! how do i do this later with batches - for now: strip first batch dim
    dmap_array = dmap_array[0, :, :, :]
    print(dmap_array.shape)
    superimposed_img = jet_heatmap * alpha + dmap_array
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    print("superimposed_img.shape ", superimposed_img.size)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))
    
    plt.imshow(dmap_array)
    plt.show()


def return_gradcam(dmap_array, heatmap, transparency):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    jet_cm = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet_cm(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)

    #resize heatmap to size of depthmap
    jet_heatmap = jet_heatmap.resize((dmap_array.shape[2], dmap_array.shape[1]))

    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    # Superimpose the heatmap on original image
    ## TODO - fix dimensions!! how do i do this later with batches - for now: strip first batch dim
    #dmap_array = dmap_array[0, :, :, :] do I need this in cgm-rg? idk
    print(dmap_array.shape)
    superimposed_img = jet_heatmap * alpha + dmap_array

    return superimposed_img







