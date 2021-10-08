# gradcam algorithm taken from F. Chollet elephant example
import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm


REPO_DIR = Path(os.getcwd()).parents[2].absolute()


def show_depthmaps(depthmaps):
    for depthmap in depthmaps:
        plt.imshow(depthmap, cmap='gray')
        plt.show()


#process multiple
def make_gradcam_heatmaps(preprocessed_depthmaps, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    heatmaps = []
    for depthmap in preprocessed_depthmaps:
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(depthmap)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
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
        heatmap = heatmap.numpy()
        heatmaps.append(heatmap)
        #plt.matshow(heatmap)
        #plt.show()
    return heatmaps


def show_heatmaps(heatmaps):
    for heatmap in heatmaps:
        plt.matshow(heatmap)
        plt.show()


#process one
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


#process multiple
def return_gradcams(preprocessed_depthmaps, heatmaps, transparency):
    superimposed_imgs = []
    i = 0
    for heatmap in heatmaps:
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        jet_cm = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet_cm(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        #resize heatmap to size of depthmap
        jet_heatmap = jet_heatmap.resize((preprocessed_depthmaps[i].shape[1], preprocessed_depthmaps[i].shape[0]))

        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        # Superimpose the heatmap on original image
        # TODO - fix dimensions!! how do i do this later with batches - for now: strip first batch dim
        superimposed_img = jet_heatmap * transparency + preprocessed_depthmaps[i]
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        superimposed_imgs.append(superimposed_img)
        #plt.imshow(superimposed_img)
        #plt.show()
        i += 1

    return superimposed_imgs


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


#process one
def return_gradcam(dmap_array, heatmap, transparency): 
    # maybe add 2 channels in depthmap? replicate depth channel 3x
    # write own overlay function 
    # see cgm-rg for depthmap vi
    # visualize depthmaps in grayscale - color is misleading
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    print("heatmap.shape = ", heatmap.shape)
    jet_cm = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet_cm(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    print("jet_heatmap.shape = ", jet_heatmap.shape)

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    print("jet_heatmap.size = ", jet_heatmap.size)

    #resize heatmap to size of depthmap
    jet_heatmap = jet_heatmap.resize((dmap_array.shape[2], dmap_array.shape[1]))
    print("jet_heatmap.size = ", jet_heatmap.size)

    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    print("jet_heatmap.shape = ", jet_heatmap.shape)
    # Superimpose the heatmap on original image
    # TODO - fix dimensions!! how do i do this later with batches - for now: strip first batch dim
    dmap_array = dmap_array[0, :, :, :]  # do I need this in cgm-rg? idk
    print("dmap_array.shape = ", dmap_array.shape)
    superimposed_img = jet_heatmap * transparency + dmap_array
    print("superimposed_img.shape ", superimposed_img.size)
    plt.imshow(superimposed_img)
    plt.show()
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    print("superimposed_img.shape ", superimposed_img.size)

    #display(Image(superimposed_img))
    #superimposed_img.show()
    plt.imshow(superimposed_img)
    plt.imshow(dmap_array)
    plt.show()

    return superimposed_img

