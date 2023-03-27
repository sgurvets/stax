#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 23:00:18 2023

@author: sgurvets
"""

import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cdist

from PIL import Image

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features

# Define image folder and output video path
image_folder = '/tmp/'
output_video = 'sorted_images_video.avi'

def get_image_dimensions(img_path):
    img = Image.open(img_path)
    return img.size

# Get the image paths and extract features
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]

image_sizes = [get_image_dimensions(img_path) for img_path in image_paths]
max_width, max_height = max([size[0] for size in image_sizes]), max([size[1] for size in image_sizes])
image_features = np.vstack([extract_features(img_path, model) for img_path in image_paths])

# Apply t-SNE and sort images by the 2D coordinates
tsne = TSNE(n_components=2, perplexity=30).fit_transform(image_features)
sorted_idx = np.lexsort((tsne[:, 1], tsne[:, 0]))
sorted_image_paths = [image_paths[i] for i in sorted_idx]

def resize_and_pad(img, target_size, pad_color=0):
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if target_ratio > img_ratio:
        new_height = target_size[1]
        new_width = int(target_size[1] * img_ratio)
    else:
        new_width = target_size[0]
        new_height = int(target_size[0] / img_ratio)

    img = img.resize((new_width, new_height), Image.ANTIALIAS)

    padded_img = Image.new('RGB', target_size, pad_color)
    padded_img.paste(img, ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2))

    return padded_img

def read_and_resize_image(img_path, target_size):
    img = Image.open(img_path)
    img = resize_and_pad(img, target_size)
    return np.array(img)


# Create the video from the sorted images
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, 2.0, max_width, max_height)

for img_path in sorted_image_paths:
    img = read_and_resize_image(img_path, (max_width, max_height))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out.write(img)

out.release()
