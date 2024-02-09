import subprocess
from PIL import Image
import os

class Cameras():
    """
    Camera class to capture images from both cameras.
    
    Args:
        resolution (tuple): resolution of the images
    
    """
    def __init__(self,
                 resolution=(224, 224),
                 directory='images',
        ):
        self.type = type
        self.resolution = resolution
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def _capture(self, camera):
        """
        Capture image from the camera.
        """
        subprocess.call(['rpicam-jpeg',
                         '-o', f'{self.directory}/camera{camera}.jpg',
                         '--width', str(self.resolution[0]),
                         '--height', str(self.resolution[1]),
                         '-t', '10',
                         '-v', '0',
                         '--camera', str(camera)])

        with open(f'{self.directory}/camera{camera}.jpg', 'rb') as f:
            img = Image.open(f)
            return img

        
    def capture(self, camera=None):
        """
        Capture images from the cameras.
        """
        if camera is None:
            img0 = self._capture(0)
            img1 = self._capture(1)
            return img0, img1
        else:
            return self._capture(camera)

        

