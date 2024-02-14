import subprocess
from PIL import Image
import os

class Camera():
    def __init__(self,
                 resolution=(224, 224),
                 camera_type='cv2'
        ):
        self.type = type
        self.resolution = resolution
        self.camera_type = camera_type
        self._init_camera()

    def _init_camera(self):
        if self.camera_type == 'cv2':
            import cv2
            self.camera = cv2.VideoCapture(0)
            self.camera.set(3, self.resolution[0])
            self.camera.set(4, self.resolution[1])

    def capture(self):
        success, frame = self.camera.read()
        if success:
            return frame
        else:
            print('Failed to capture image')
            return None

class Cameras():
    """
    Camera class to capture images from both cameras.
    
    Args:
        resolution (tuple): resolution of the images
    
    """
    def __init__(self,
                 resolution=(224, 224),
                 directory='images',
                 camera_type='cv2'
        ):
        self.type = type
        self.resolution = resolution
        self.directory = directory
        self.camera_type = camera_type
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

        img = Image.open(f'{self.directory}/camera{camera}.jpg')
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

        

