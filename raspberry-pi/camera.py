import cv2

class Camera():
    """
    Camera class to capture images from both cameras.
    
    Args:
        resolution (tuple): resolution of the images
    
    """
    def __init__(self,
                 resolution=(224, 224),
        ):
        self.type = type
        self.resolution = resolution
        self.cameras = self._initialize_cameras()

    def _initialize_cameras(self):
        """
        Initialize the cameras.
        """

        cameras = []
        for idx in range(2):
            camera = cv2.VideoCapture(idx)
            camera.set(3, self.resolution[0])
            camera.set(4, self.resolution[1])
            cameras.append(camera)
        return cameras
        
    def capture(self):
        """
        Capture images from both cameras.
        """
        images = []
        for camera in self.cameras:
            ret, frame = camera.read()
            images.append(frame)
        return images

