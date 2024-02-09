import numpy as np
import os
import subprocess
import time

class Generator():
    def __init__(self, cube, save_dir):
        self.cube = cube
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    @property
    def num_examples(self):
        files = [f for f in os.listdir(self.save_dir) if f.endswith('.jpg')]
        return len(files)

    def random_fine_moves(self, max_moves=6):
        steps = []
        for i in range(6):
            if np.random.rand() > 0.5:
                n = np.random.randint(-max_moves, max_moves)
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

    def generate(self, examples=100, scramble=2):
        for i in range(examples):
            self.cube.scramble(2)
            time.sleep(0.5)
            steps = self.random_fine_moves()
            time.sleep(0.5)
            example_num = self.num_examples
            self.save_picture(example_num)
            self.save_moves(steps, example_num)
            self.revert_moves(steps)
            time.sleep(0.5)

    def save_picture(self, example_num):
        subprocess.call(['rpicam-jpeg', '-o', self.save_dir + f'/{example_num}.jpg', '-t', '10', '--width', '480', '--height', '480', '-v', '0', '--camera', '1'])

    def save_moves(self, steps, example_num):
        np.save(self.save_dir + f'/{example_num}.npy', steps)