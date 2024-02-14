from server.model import PoseEstimator
import torch

class PoseAligner(PoseEstimator):
    def __init__(self,
                 cube,
                 camera,
                 checkpoint_path,):
        super().__init__()
        self.cube = cube
        self.camera = camera
        self.checkpoint_path = checkpoint_path
        self._load_checkpoint()

    def _load_checkpoint(self):
        print(f'Loading checkpoint from {self.checkpoint_path}')
        self.load_state_dict(torch.load(checkpoint_path))
        print('Checkpoint loaded')

    def _capture_and_estimate_pose(self):
        image = self.camera.capture()
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0) / 255
        estimated_pose = self(image)
        estimated_pose = estimated_pose.long().item()
        return estimated_pose

    def _move_cube(self, pose):
        for i in range(6):
            if pose[i] > 0:
                move = cube.moves_list[i]

            elif pose[i] < 0:
                move = cube.moves_list[i + 1]

            else:
                move = None
                
            if move:
                cube.fine_move(move, steps=abs(pose[i]))

    def align(self):
        aligned = False
        while True:
            estimated_pose = self._capture_and_estimate_pose()
            if sum(estimated_pose) == 0:
                break
            else:
                self._move_cube(estimated_pose)
        
if __name__ == '__main__':
    from cube import Cube
    from camera import Cameras
    cube = Cube()
    cameras = Cameras()
    pose_aligner = PoseAligner(cube, cameras, 'PoseEstimator_resolute-ring-2_best.pth')