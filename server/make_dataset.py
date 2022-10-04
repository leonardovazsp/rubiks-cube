'''
Module to create the dataset.
This module access the rubiks cube API to rotate the cube and get the images and labels.
The images are stored in the `data` directory, split into train, test and val sets.
The inputs for the model are the images (jpg), and the labels are numpy files of tensor of shape (27, 6).
The labels are the one-hot encoded representation of the cube state (colors).
'''

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import requests
import urllib.request
from PIL import Image
from tqdm import tqdm

# Load config
with open('config.json') as f:
    config = json.load(f)

# Create the data directory
if not os.path.exists(config['data_dir']):
    os.makedirs(config['data_dir'])

# Create the train, test and val directories
for split in config['splits']:
    if not os.path.exists(os.path.join(config['data_dir'], split)):
        os.makedirs(os.path.join(config['data_dir'], split))

# Create the images and labels directories
for split in config['splits']:
    if not os.path.exists(os.path.join(config['data_dir'], split, 'images')):
        os.makedirs(os.path.join(config['data_dir'], split, 'images'))
    if not os.path.exists(os.path.join(config['data_dir'], split, 'labels')):
        os.makedirs(os.path.join(config['data_dir'], split, 'labels'))

# Create the dataset
for split in config['splits']:
    print(f'Creating {split} dataset...')
    for i in tqdm(range(config['num_samples'])):
        # Get the image and label
        response = requests.get(config['api_url'])
        data = response.json()
        image = data['image']
        label = data['label']

        # Save the image
        image_name = f'{i}.jpg'
        image_path = os.path.join(config['data_dir'], split, 'images', image_name)
        urllib.request.urlretrieve(image, image_path)

        # Save the label
        label_name = f'{i}.npy'
        label_path = os.path.join(config['data_dir'], split, 'labels', label_name)
        np.save(label_path, label)