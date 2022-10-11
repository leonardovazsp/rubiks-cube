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
import pickle
import copy
import torch
import argparse

# Load config
with open('config.json') as f:
    config = json.load(f)

# Create the data directory
if not os.path.exists(config['data_dir']):
    os.makedirs(config['data_dir'])

# Create the train, test and val directories
for split in config['splits_ratio'].keys():
    if not os.path.exists(os.path.join(config['data_dir'], split)):
        os.makedirs(os.path.join(config['data_dir'], split))

# Create the images and labels directories
for split in config['splits_ratio'].keys():
    if not os.path.exists(os.path.join(config['data_dir'], split, 'images')):
        os.makedirs(os.path.join(config['data_dir'], split, 'images'))
    if not os.path.exists(os.path.join(config['data_dir'], split, 'labels')):
        os.makedirs(os.path.join(config['data_dir'], split, 'labels'))

class SampleGenerator:
    def __init__(self):
        self.url = config['api_url']
        self.data_dir = config['data_dir']
        self.splits_ratio = config['splits_ratio']
        self.splits = sum([[split] * int(ratio * 100) for split, ratio in self.splits_ratio.items()], [])
        self.moves_list = ['right', 'left', 'top', 'bottom', 'front', 'back']

    def process_label(self, label):
        label_1 = label[[0, 1, 4]]
        label_2 = label[[2, 3, 5]]
        label_2_ = copy.deepcopy(label_2)
        label_2[0] = np.rot90(label_2_[1], 2)
        label_2[1] = np.rot90(label_2_[0], 2)
        label_2[2] = np.rot90(label_2_[2], 1)
        return label_1, label_2
    
    def send_request(self, filename=None, move=None):
        # Send a request to the API to rotate the cube and get the image and label
        # Two images and labels are received via the API via pickle
        # If move is None, rotate the cube randomly otherwise rotate the cube according to the move
        if move is None:
            move = random.choice(self.moves_list)
        r = requests.post(self.url, data=move)
        if filename is not None:
            filename = os.path.join(self.data_dir, filename)
            img_1, img_2, label = pickle.loads(r.content)
            label_1, label_2 = self.process_label(label)
            np.save(filename + '_1.npy', img_1[:, :, ::-1])
            np.save(filename + '_2.npy', img_2[:, :, ::-1])
            np.save(filename.replace('images', 'labels') + '_1.npy', label_1)
            np.save(filename.replace('images', 'labels') + '_2.npy', label_2)

    def get_imgs(self):
        # Get get the images and labels from the API without moving the cube
        r = requests.post(self.url, data='get_imgs')
        img_1, img_2, label = pickle.loads(r.content)
        return img_1[:, :, ::-1], img_2[:, :, ::-1], label

    def generate_dataset(self, num_images=100, reset=True):
        # Generate the dataset
        # If reset is True, send a final request to the API to reset the cube
        for i in tqdm(range(num_images)):
            split = random.choice(self.splits)
            base_count = len(os.listdir(os.path.join(self.data_dir, split, 'images')))//2
            filename = os.path.join(split, 'images', str(base_count))
            self.send_request(filename=filename)
        if reset:
            self.send_request(move='reset')

class Dataset:
    def __init__(self, resolution, data_dir, split, background_augmentation=0):
        self.resolution = resolution # (width, height)
        self.background_augmentation = background_augmentation
        self.data_dir = data_dir
        self.split = split
        self.images = os.listdir(os.path.join(self.data_dir, self.split, 'images'))
        self.labels = os.listdir(os.path.join(self.data_dir, self.split, 'labels'))
        self.mask1 = np.array(Image.open(os.path.join(self.data_dir, 'mask1.png'))).max(axis=-1, keepdims=True)/255
        self.mask2 = np.array(Image.open(os.path.join(self.data_dir, 'mask2.png'))).max(axis=-1, keepdims=True)/255
        self.backgrounds = os.listdir(os.path.join(self.data_dir, 'backgrounds'))

    def __len__(self):
        return len(self.images)

    def augment(self, image, mask):
        # image = image[40:-40,:,:]  # Crop center from (320, 240) to (240, 240)
        if np.random.rand(1)[0] < self.background_augmentation:
            background = random.choice(self.backgrounds)
            background = Image.open(os.path.join(self.data_dir, 'backgrounds', background))
            background = background.rotate(random.randint(0, 360), expand=True)
            background = background.transpose(Image.FLIP_LEFT_RIGHT) if np.random.random() < 0.5 else background
            background = background.resize(image.shape[:2][::-1])
            background = np.array(background) / 255.
            image = image/255.
            image = image * mask + background * (1 - mask)
            image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image = Image.fromarray(image)
        image = image.resize(self.resolution)
        image = image.rotate(random.randint(-40, 40))
        image = np.array(image)/255
        image = np.clip(image * random.uniform(0.5, 2.5), 0, 1)
        return image.transpose(2, 0, 1)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.data_dir, self.split, 'images', self.images[idx]))
        mask = self.mask1 if self.images[idx][-5] == 1 else self.mask2
        image = self.augment(image, mask)
        label = np.load(os.path.join(self.data_dir, self.split, 'labels', self.labels[idx]))
        label = label.reshape(-1)
        label = np.eye(6)[label]
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        return image, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-images', type=int, default=0, help='Number of training images')
    args = parser.parse_args()
    dataset_generator = SampleGenerator()
    dataset_generator.generate_dataset(num_images=args.num_images)