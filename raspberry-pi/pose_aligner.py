import sys
sys.path.append("/home/pi/rubiks-cube")
from server.model import PoseEstimator
import torch
from camera import Camera
import time
import numpy as np

class PoseAligner(PoseEstimator):
    def __init__(self,
                 cube,
                 checkpoint_path,
                 device='cpu'):
        super().__init__(fc_sizes = [1024, 2048, 1024])
        self.cube = cube
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.to(device)
        self._load_checkpoint()
        self.camera = Camera(self.input_shape[1:])

    def _load_checkpoint(self):
        print(f'Loading checkpoint from {self.checkpoint_path}')
        self.load_state_dict(torch.load(self.checkpoint_path, map_location=torch.device(self.device)))
        print('Checkpoint loaded')

    def _capture_and_estimate_pose(self):
        self.train()
        image = self.camera.capture()
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0) / 255
        estimated_pose = self(image)
        estimated_pose = estimated_pose > 0.4
        return estimated_pose.long().squeeze().tolist()

    def _move_cube(self, pose):
        print(f'Moves: {self.cube.moves_list}')
        for i in range(12):
            if pose[i] == 1:
                if i%2 == 0:
                    self.cube.fine_move(self.cube.moves_list[i+1], 1)
                    print(f'Moving {self.cube.moves_list[i+1]}')
                else:
                    self.cube.fine_move(self.cube.moves_list[i-1], 1)
                    print(f'Moving {self.cube.moves_list[i-1]}')

    def align(self):
        aligned = False
        while True:
            estimated_pose = self._capture_and_estimate_pose()
            print(f'Estimated pose: {estimated_pose}')
            if sum(np.abs(estimated_pose)) == 0:
                break
            else:
                self._move_cube(estimated_pose)
        
if __name__ == '__main__':
    from cube import Cube
    cube = Cube()
    cube.driver.set_delay(900)
    cube.driver.activate()
    pose_aligner = PoseAligner(cube, 'PoseEstimator_radiant-dumpling-20_best.pth')