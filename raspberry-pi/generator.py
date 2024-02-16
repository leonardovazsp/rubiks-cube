import numpy as np
import os
import subprocess
import time
from camera import Camera
import cv2

class Generator():
    def __init__(self, cube, save_dir, resolution=(480, 480), examples_offset=0):
        self.cube = cube
        self.save_dir = save_dir
        self.camera = Camera(resolution)
        self.examples_offset = examples_offset
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    @property
    def num_examples(self):
        files = [f for f in os.listdir(self.save_dir) if f.endswith('.jpg')]
        return len(files) + self.examples_offset

    def random_fine_moves(self, max_moves=6, min_moves=2):
        steps = []
        for i in range(6):
            if np.random.rand() > 0.5:
                n = np.random.randint(-max_moves, max_moves)
                if n < 0:
                    n = min(n, -min_moves)
                else:
                    n = max(n, min_moves)
                steps.append(n)
            else:
                steps.append(0)

        for step, move in zip(steps, self.cube.moves_list[::2]):
            if step < 0:
                self.cube.fine_move(move + '_rev', -step)
            elif step > 0:
                self.cube.fine_move(move, step)

        return steps

    def revert_moves(self, steps):
        for step, move in zip(steps, self.cube.moves_list[::2]):
            if step < 0:
                self.cube.fine_move(move, -step)
            elif step > 0:
                self.cube.fine_move(move + '_rev', step)

    def generate(self, examples=100, scramble=1):
        for i in range(examples):
            self.cube.scramble(scramble)
            time.sleep(0.5)
            steps = self.random_fine_moves()
            time.sleep(0.5)
            example_num = self.num_examples
            # self.save_picture(example_num)
            frame = self.camera.capture()
            self._save_frame(frame)
            self.save_moves(steps, example_num)
            self.revert_moves(steps)
            time.sleep(0.5)

    def _save_frame(self, frame):
        cv2.imwrite(self.save_dir + f'/{self.num_examples}.jpg', frame)

    def save_picture(self, example_num):
        subprocess.call(['rpicam-jpeg', '-o', self.save_dir + f'/{example_num}.jpg', '-t', '10', '--width', '480', '--height', '480', '-v', '0', '--camera', '0'])

    def save_moves(self, steps, example_num):
        np.save(self.save_dir + f'/{example_num}.npy', steps)