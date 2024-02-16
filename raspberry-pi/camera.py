import subprocess
from PIL import Image
import os
import cv2
import threading
import queue
# import signal
# import sys

class Camera():
    def __init__(self,
                 resolution=(224, 224),
                 src=0,
        ):
        self.resolution = resolution
        self.q = queue.Queue()
        self.running = True
        self.cap = self._init_camera(src)
        self.thread = threading.Thread(target=self._reader)
        self.thread.start()
        
    def __del__(self):
        self.release()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def _init_camera(self, src):
        cap = cv2.VideoCapture(src)
        cap.set(3, self.resolution[0])
        cap.set(4, self.resolution[1])
        return cap

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def capture(self):
        if self.q.empty():
            return None
        return self.q.get()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

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

        

