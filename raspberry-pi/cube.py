'''
This module provides an API to interact with the Rubik's Cube.
It also provides a virtual cube to keep track of the cube state.
The cube state is represented as a tensor of shape (54, 6).
The state is one-hot encoded, where each element is a vector of length 6,
which represents the color of the cubelet.
The state is stored in a file called `state.npy`.
'''

import cube_driver as cd
import numpy as np
import os
import random

class Cube:
    def __init__(self):
        # Load the cube state and history (create a new files if not found)
        self.state = self.load_state()
        self.history = self.load_history()
        self.moves_list = ['right', 'right_rev', 'left', 'left_rev', 'top', 'top_rev', 'bottom', 'bottom_rev', 'front', 'front_rev', 'back', 'back_rev']
        self.moves = [self.right, self.right_rev, self.left, self.left_rev, self.top, self.top_rev, self.bottom, self.bottom_rev, self.front, self.front_rev, self.back, self.back_rev]

    def load_state(self):
        # Load the cube state
        if os.path.exists('state.npy'):
            state = np.load('state.npy')
        else:
            state = np.arange(54) // 9 # 54 cubelets, 9 cubelets per face
            state = np.eye(6)[state] # one-hot encode
            np.save('state.npy', state) # save the state
        return state

    def load_history(self):
        # Load the history
        if os.path.exists('history.npy'):
            history = np.load('history.npy')
        else:
            history = np.array([])
            np.save('history.npy', history)

    def reset(self):
        # Reset the cube
        for move in reversed(self.history):
            if move[-4:] == '_rev':
                move = move[:-4]
            else:
                move = move + '_rev'
            self.moves[self.moves_list.index(move)]()
        self.state = np.arange(54) // 9
        self.state = np.eye(6)[self.state]
        np.save('state.npy', self.state)
        self.history = np.array([])
        np.save('history.npy', self.history)
    
    


