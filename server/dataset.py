import os
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms

if not os.path.exists('data'):
    os.mkdir('data')

class Dataset():
    """
    Dataset class for the Rubik's Cube.

    """
    def __init__(self, resolution=(96, 96), model_type='color_recognition', directory='data'):
        self.resolution = resolution
        self.masks = None
        self.backgrounds = None
        self.type = model_type
        self.cache = {}
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.directory = directory
        if not os.path.exists(os.path.join(directory, model_type)):
            os.mkdir(os.path.join(directory, model_type))
    
    @property
    def images_per_example(self):
        if self.type == 'color_recognition':
            return 2
        elif self.type == 'pose_estimation':
            return 1

    @property
    def size(self):
        """
        Returns the size of the dataset.
        """
        files = [f for f in os.listdir(os.path.join(self.directory, self.type)) if f.endswith('.npy')]
        return len(files)

    @property
    def config(self):
        return {
            'resolution': self.resolution,
            'backgrounds': self.backgrounds is not None,
        }

    def __len__(self):
        return self.size

    def set_masks(self, path):
        """
        Sets the masks for each camera.

        Args:
            path: path to the masks
        """
        self.masks = []
        try:
            for i in range(self.images_per_example):
                with open(f'{path}/{i}.jpg', 'rb') as f:
                    self.masks.append(Image.open(f))
        except FileNotFoundError:
            print(f'No masks found. Masks should be named 0.jpg, 1.jpg, and located in the specified folder {path}')

    def set_backgrounds(self, path):
        """
        Sets path for background images."""
        self.backgrounds = os.listdir(path)

    def save_example(self, images, state):
        """
        Saves an example to the data folder.

        Args:
            images: list of 2 images
            state: numpy array of the cube state
        """
        example_num = self.size
        for i, image in enumerate(images):
            image.save(os.path.join(self.directory, f'{self.type}/{example_num}_{i}.jpg'))
            # image.save(f'{self.type}/{example_num}_{i}.jpg')

        # with open(f'{self.type}/{example_num}.npy', 'wb') as f:
        with open(os.path.join(self.directory, f'{self.type}/{example_num}.npy'), 'wb') as f:
            np.save(f, state)

    def _load_example(self, example_num):
        """
        Loads an example from the data folder.

        Returns:
            images: list of 2 images
            state: numpy array of the cube state
        """
        images = []
        for i in range(self.images_per_example):
            images.append(Image.open(os.path.join(self.directory, f'{self.type}/{example_num}_{i}.jpg')))
            # images.append(Image.open(f'/mnt/{self.type}/{example_num}_{i}.jpg'))

        # with open(f'/mnt/{self.type}/{example_num}.npy', 'rb') as f:
        with open(os.path.join(self.directory, f'{self.type}/{example_num}.npy'), 'rb') as f:
            state = np.load(f)

        return images, state

    def _add_background(self, images):
        """
        Cut out the cube from the images and paste it on a random background.

        Args:
            images: list of 2 images
            masks: list of 2 masks
            background: background image

        Returns:
            images: list of 2 images with background
        """
        background = Image.open(f'backgrounds/{random.choice(self.backgrounds)}')
        background = background.rotate(random.randint(0, 360), expand=True)
        background = background.transpose(Image.FLIP_LEFT_RIGHT) if np.random.random() < 0.5 else background
        background = background.resize(images[0].shape[:2][::-1])
        background = np.array(background) / 255.
        output = []
        for img, mask in zip(images, self.masks):
            img = img/255.
            img = img * mask + background * (1 - mask)
            img = Image.fromarray((img * 255).astype(np.uint8))
            output.append(img)
        return output

    def _center_crop(self, images, ratio=0.8):
        """
        Center crop the images.

        Args:
            images: list of 2 images

        Returns:
            images: list of 2 images center cropped
        """
        output = []
        min_dim = min(images[0].size)
        crop_size = (int(min_dim * ratio), int(min_dim * ratio))
        crop_size = (crop_size[0] - crop_size[0] % 2, crop_size[1] - crop_size[1] % 2)
        for img in images:
            img = transforms.CenterCrop(crop_size)(img)
            output.append(img)
        return output


    def _rotate(self, images, angle):
        """
        Rotate the images by a random angle.

        Args:
            images: list of 2 images
            angle: maximum angle to rotate by

        Returns:
            images: list of 2 images rotated by angle
        """

        angle = np.random.randint(-angle, angle)
        output = []
        for img in images:
            img = img.rotate(angle)
            output.append(img)
        return output

    def _resize(self, images):
        """
        Resize the images to self.resolution.

        Args:
            images: list of 2 images

        Returns:
            images: list of 2 images resized to self.resolution
        """
        output = []

        for img in images:
            img = img.resize(self.resolution)
            output.append(img)
        return output

    def _augment(self, images, angle=30, chance=0.5):
        """
        Augment the images.

        Args:
            images: list of 2 images
            masks: list of 2 masks
            background: background image

        Returns:
            images: list of 2 images augmented
        """
        if self.masks and self.backgrounds and np.random.rand() < chance:
            images = self._add_background(images)

        images = self._rotate(images, angle) if np.random.rand() < chance else images
        images = self._center_crop(images)
        images = self._resize(images)
        
        return images

    def __getitem__(self, idx):
        """
        Returns an example from the dataset.

        Args:
            idx: index of the example

        Returns:
            images: list of 2 images
            state: numpy array of the cube state
        """
        if idx in self.cache.keys():
            print('Cache hit!')
            return self.cache[idx]
            
        if idx >= self.size:
            raise IndexError(f'Index {idx} out of range for dataset of size {self.size}')
        
        images, state = self._load_example(idx)
        images = self._augment(images, chance=0.5)
        images = [torch.tensor(np.array(img).transpose(2, 0, 1)).float() for img in images]
        state = torch.tensor(state).float()
        if self.type == 'pose_estimation':
            state = state.unsqueeze(0)
        self.cache[idx] = (images, state)
        return images, state