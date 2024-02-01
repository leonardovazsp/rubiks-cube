import os
from PIL import Image

if not os.path.exists('data'):
    os.mkdir('data')

class Dataset():
    def __init__(self, resolution=(224, 224)):
        self.resolution = resolution
        self.masks = None
        self.backgrounds = None

    @property
    def size(self):
        """
        Returns the size of the dataset.
        """
        total_files = len(os.listdir('data'))
        total_examples = len(os.listdir('data')) // 3 # 3 files per example
        return total_examples

    def set_masks(self, path):
        """
        Sets the masks for each camera.

        Args:
            path: path to the masks
        """
        self.masks = []
        for i in range(2):
            with open(f'{path}/{i}.jpg', 'rb') as f:
                self.masks.append(Image.open(f))

    def set_backgrounds(self, path):
        """
        Sets path for background images."""
        self.backgrounds = os.listdir(path)

    def save_example(self, images, state):
        """
        Saves an example to the data folder.
        """
        example_num = self.size
        for i, image in enumerate(images):
            with open(f'data/{example_num}_{i}.jpg', 'wb') as f:
                f.write(image)

        with open(f'data/{example_num}.npy', 'wb') as f:
            np.save(f, state)

    def _load_example(self, example_num):
        """
        Loads an example from the data folder.

        Returns:
            images: list of 2 images
            state: numpy array of the cube state
        """
        images = []
        for i in range(2):
            with open(f'data/{example_num}_{i}.jpg', 'rb') as f:
                images.append(Image.open(f))

        with open(f'data/{example_num}.npy', 'rb') as f:
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
        for img, mask in zip(images, masks):
            img = img/255.
            img = img * mask + background * (1 - mask)
            img = Image.fromarray((img * 255).astype(np.uint8))
            output.append(img)
        return output

    def _rotate(self, images, angle=40):
        """
        Rotate the images by a random angle.

        Args:
            images: list of 2 images
            angle: maximum angle to rotate by

        Returns:
            images: list of 2 images rotated by angle
        """

        angle = random.randint(-angle, angle)
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

    def _augment(self, images, angle=40, chance=0.5):
        """
        Augment the images.

        Args:
            images: list of 2 images
            masks: list of 2 masks
            background: background image

        Returns:
            images: list of 2 images augmented
        """
        if self.masks and self.backgrounds and np.random.rand() > chance:
            images = self._add_background(images)

        images = self._resize(images) if np.random.rand() > chance else images
        images = self._rotate(images, angle) if np.random.rand() > chance else images
        
        return images

    def __get_item__(self, idx):
        """
        Returns an example from the dataset.

        Args:
            idx: index of the example

        Returns:
            images: list of 2 images
            state: numpy array of the cube state
        """
        images, state = self._load_example(idx)
        images = self._augment(images)
        return images, state