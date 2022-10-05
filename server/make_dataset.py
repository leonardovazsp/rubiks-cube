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

class DatasetGenerator:
    # Class to retrieve the images and labels from the API
    # Send a request to the API to rotate the cube and get the image and label
    def __init__(self):
        self.url = config['api_url']
        self.data_dir = config['data_dir']
        self.splits_ratio = config['splits_ratio']
        self.splits = sum([[split] * int(ratio * 100) for split, ratio in splits_ratio.items()], [])
        self.moves_list = ['right', 'left', 'top', 'bottom', 'front', 'back']
    
    def send_request(self, filename=None, move=None):
        # Send a request to the API to rotate the cube and get the image and label
        # Two images and labels are received via the API in a json file
        # If filename is not None, save the image to the file
        # If move is None, rotate the cube randomly
        # If move is not None, rotate the cube according to the move
        if move is None:
            move = random.choice(self.moves_list)
        r = requests.post(self.post_url, data=move)
        if filename is not None:
            filename = os.path.join(self.data_dir, filename)
            cube_state_1 = r.json()['cube_state_1']
            cube_state_2 = r.json()['cube_state_2']
            image_1 = r.json()['image_1']
            image_2 = r.json()['image_2']
            urllib.request.urlretrieve(image_1, filename + '_1.jpg')
            urllib.request.urlretrieve(image_2, filename + '_2.jpg')
            np.save(filename.replace('images', 'labels') + '_1.npy', cube_state_1)
            np.save(filename.replace('images', 'labels') + '_2.npy', cube_state_2)

    def generate_dataset(self, num_images=100, reset=True):
        # Generate the dataset
        # If reset is True, send a final request to the API to reset the cube
        for i in tqdm(range(num_images)):
            split = random.choice(self.splits)
            base_count = len(os.listdir(os.path.join(self.data_dir, split, 'images')))
            filename = os.path.join(split, 'images', str(base_count))
            self.send_request(filename=filename)





splits_ratio = {'train': 0.8, 'test': 0.1, 'val': 0.1}
splits = [s for s in [[split] * ratio * 100 for split, ratio in splits_ratio.items()]]


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