'''
This module provides an API to interact with the Rubik's Cube.
It also provides a virtual cube to keep track of the cube state.
The state is stored in a file called `state.npy`.
'''

import numpy as np
import os
import random
import json
import copy
import inspect
from driver import Driver, COMMANDS


class Cube:
    def __init__(self):
        # Load the cube state and history (create a new files if not found)
        self.state = self.load_state()
        self.history = self.load_history()
        self.moves_list = list(COMMANDS.keys())
        self.driver = Driver()

    def load_state(self):
        # Load the cube state
        if os.path.exists('state.npy'):
            state = np.load('state.npy')
        else:
            state = np.arange(54) // 9 # 54 cubelets, 9 cubelets per face
            state = state.reshape(6, 3, 3) # reshape to 6 faces, 3x3 cubelets
            np.save('state.npy', state) # save the state
        return state

    def load_history(self):
        # Load the history
        if os.path.exists('history.npy'):
            history = np.load('history.npy')
        else:
            history = np.array([])
            np.save('history.npy', history)
        return list(history)

    def reset(self):
        # Reset the cube
        for move in reversed(self.history):
            if move[-4:] == '_rev':
                move = move[:-4]
            else:
                move = move + '_rev'
            assert move in self.moves_list, f"move must belong to the set of moves {self.moves_list}"
            getattr(self, move)()

        self.state = np.arange(54) // 9
        self.state = self.state.reshape(6, 3, 3)
        np.save('state.npy', self.state)
        self.history = list(np.array([]))
        np.save('history.npy', self.history)

    def get_cube_state(self):
        # Get the cube state
        return self.state

    def fine_move(self, move, steps=1):
        self.driver.fine_move(move, steps=steps)

    def scramble(self, n):
        # Scramble the cube
        available_moves = copy.deepcopy(self.moves_list)
        last_move = None
        for i in range(n):
            if last_move:
                if last_move[-4:] == '_rev':
                    available_moves.remove(last_move[:-4])
                else:
                    available_moves.remove(last_move + '_rev')
            move = random.choice(available_moves)
            last_move = move
            available_moves = copy.deepcopy(self.moves_list)
            getattr(self, move)()
            
    def top(self):
        # Rotate the top face clockwise
        self.driver.move_cube('top')
        temp_state = copy.deepcopy(self.state)
        self.state[0, 0] = temp_state[1, 0]
        self.state[1, 0] = temp_state[2, 0]
        self.state[2, 0] = temp_state[3, 0]
        self.state[3, 0] = temp_state[0, 0]
        self.state[4, :] = np.rot90(temp_state[4, :], -1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)

    def top_rev(self):
        # Rotate the top face counter-clockwise
        self.driver.move_cube('top_rev')
        temp_state = copy.deepcopy(self.state)
        self.state[0, 0] = temp_state[3, 0]
        self.state[1, 0] = temp_state[0, 0]
        self.state[2, 0] = temp_state[1, 0]
        self.state[3, 0] = temp_state[2, 0]
        self.state[4, :] = np.rot90(temp_state[4, :], 1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)

    def right(self):
        # Rotate the right face clockwise
        self.driver.move_cube('right')
        temp_state = copy.deepcopy(self.state)
        self.state[0, :, 2] = temp_state[5, :, 2]
        self.state[4, :, 2] = temp_state[0, :, 2]
        self.state[2, :, 0] = np.flip(temp_state[4, :, 2], 0)
        self.state[5, :, 2] = np.flip(temp_state[2, :, 0], 0)
        self.state[1, :] = np.rot90(temp_state[1, :], -1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)

    def right_rev(self):
        # Rotate the right face counter-clockwise
        self.driver.move_cube('right_rev')
        temp_state = copy.deepcopy(self.state)
        self.state[0, :, 2] = temp_state[4, :, 2]
        self.state[4, :, 2] = np.flip(temp_state[2, :, 0], 0)
        self.state[2, :, 0] = np.flip(temp_state[5, :, 2], 0)
        self.state[5, :, 2] = temp_state[0, :, 2]
        self.state[1, :] = np.rot90(temp_state[1, :], 1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)
    
    def left(self):
        # Rotate the left face clockwise
        self.driver.move_cube('left')
        temp_state = copy.deepcopy(self.state)
        self.state[0, :, 0] = temp_state[4, :, 0]
        self.state[4, :, 0] = np.flip(temp_state[2, :, 2], 0)
        self.state[2, :, 2] = np.flip(temp_state[5, :, 0], 0)
        self.state[5, :, 0] = temp_state[0, :, 0]
        self.state[3, :] = np.rot90(temp_state[3, :], -1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)
              

    def left_rev(self):
        # Rotate the left face counter-clockwise
        self.driver.move_cube('left_rev')
        temp_state = copy.deepcopy(self.state)
        self.state[0, :, 0] = temp_state[5, :, 0]
        self.state[4, :, 0] = temp_state[0, :, 0]
        self.state[2, :, 2] = np.flip(temp_state[4, :, 0], 0)
        self.state[5, :, 0] = np.flip(temp_state[2, :, 2], 0)
        self.state[3, :] = np.rot90(temp_state[3, :], 1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)

    def bottom(self):
        # Rotate the bottom face clockwise
        self.driver.move_cube('bottom')
        temp_state = copy.deepcopy(self.state)
        self.state[0, 2] = temp_state[3, 2]
        self.state[3, 2] = temp_state[2, 2]
        self.state[2, 2] = temp_state[1, 2]
        self.state[1, 2] = temp_state[0, 2]
        self.state[5, :] = np.rot90(temp_state[5, :], -1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)

    def bottom_rev(self):
        # Rotate the bottom face counter-clockwise
        self.driver.move_cube('bottom_rev')
        temp_state = copy.deepcopy(self.state)
        self.state[0, 2] = temp_state[1, 2]
        self.state[3, 2] = temp_state[0, 2]
        self.state[2, 2] = temp_state[3, 2]
        self.state[1, 2] = temp_state[2, 2]
        self.state[5, :] = np.rot90(temp_state[5, :], 1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)

    def back(self):
        # Rotate the back face clockwise
        self.driver.move_cube('back')
        temp_state = copy.deepcopy(self.state)
        self.state[1, :, 2] = np.flip(temp_state[5, 2, :], 0)
        self.state[5, 2, :] = temp_state[3, :, 0]
        self.state[3, :, 0] = np.flip(temp_state[4, 0, :], 0)
        self.state[4, 0, :] = temp_state[1, :, 2]
        self.state[2, :] = np.rot90(temp_state[2, :], -1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)

    def back_rev(self):
        # Rotate the back face counter-clockwise
        self.driver.move_cube('back_rev')
        temp_state = copy.deepcopy(self.state)
        self.state[1, :, 2] = temp_state[4, 0, :]
        self.state[5, 2, :] = np.flip(temp_state[1, :, 2], 0)
        self.state[3, :, 0] = temp_state[5, 2, :]
        self.state[4, 0, :] = np.flip(temp_state[3, :, 0], 0)
        self.state[2, :] = np.rot90(temp_state[2, :], 1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)

    def front(self):
        # Rotate the front face clockwise
        self.driver.move_cube('front')
        temp_state = copy.deepcopy(self.state)
        self.state[1, :, 0] = temp_state[4, 2, :]
        self.state[4, 2, :] = np.flip(temp_state[3, :, 2], 0)
        self.state[3, :, 2] = temp_state[5, 0, :]
        self.state[5, 0, :] = np.flip(temp_state[1, :, 0], 0)
        self.state[0, :] = np.rot90(temp_state[0, :], -1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)

    def front_rev(self):
        # Rotate the front face counter-clockwise
        self.driver.move_cube('front_rev')
        temp_state = copy.deepcopy(self.state)
        self.state[1, :, 0] = np.flip(temp_state[5, 0, :], 0)
        self.state[4, 2, :] = temp_state[1, :, 0]
        self.state[3, :, 2] = np.flip(temp_state[4, 2, :], 0)
        self.state[5, 0, :] = temp_state[3, :, 2]
        self.state[0, :] = np.rot90(temp_state[0, :], 1)
        self.history.append(inspect.currentframe().f_code.co_name)
        np.save('history.npy', self.history)

if __name__ == '__main__':
    import time
    cube = Cube()
    time.sleep(5)
    for i in range(10):
        cube.top()
        cube.front()
        cube.left()
        cube.right()
        cube.bottom()
        cube.back()
        cube.reset()
        time.sleep(1)
    print(cube.state)

