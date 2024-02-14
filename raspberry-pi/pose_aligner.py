import sys
sys.path.append("/home/pi/rubiks-cube")
from server.model import PoseEstimator
import torch
from camera import Camera

class PoseAligner(PoseEstimator):
    def __init__(self,
                 cube,
                 checkpoint_path,
                 device='cpu'):
        super().__init__()
        self.cube = cube
        self.camera = Camera(self.input_shape[1:])
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.to(device)
        self._load_checkpoint()

    def _load_checkpoint(self):
        print(f'Loading checkpoint from {self.checkpoint_path}')
        self.load_state_dict(torch.load(self.checkpoint_path, map_location=torch.device(self.device)))
        print('Checkpoint loaded')

    def _capture_and_estimate_pose(self):
        image = self.camera.capture()
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0) / 255
        estimated_pose = self(image)
        estimated_pose = estimated_pose.long().squeeze().tolist()
        return estimated_pose

    def _move_cube(self, pose):
        for i in range(6):
            if pose[i] < 0:
                move = cube.moves_list[i]

            elif pose[i] > 0:
                move = cube.moves_list[i + 1]

            else:
                move = None
                
            if move:
                cube.fine_move(move, steps=abs(pose[i]))

    def align(self):
        aligned = False
        while True:
            estimated_pose = self._capture_and_estimate_pose()
            print(f'Estimated pose: {estimated_pose}')
            if sum(estimated_pose) == 0:
                break
            else:
                self._move_cube(estimated_pose)
        
if __name__ == '__main__':
    from cube import Cube
    cube = Cube()
    pose_aligner = PoseAligner(cube, 'PoseEstimator_exquisite-valentine-5_best.pth')