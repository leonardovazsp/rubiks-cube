import time
import numpy as np
import copy
import pickle
import inspect
import random

class Cube():
    def __init__(self, state=None):
        if state is None:
            self.state = self.get_solved_cube()
        else:
            self.state = state
        self.solved_cube = self.get_solved_cube()
        self.moves_list = ['right', 'right_rev', 'left', 'left_rev', 'top', 'top_rev', 'bottom', 'bottom_rev', 'front', 'front_rev', 'back', 'back_rev']
        self.moves = [self.right, self.right_rev, self.left, self.left_rev, self.top, self.top_rev, self.bottom, self.bottom_rev, self.front, self.front_rev, self.back, self.back_rev]
        
    def step(self, move_idx):
            move = self.moves[move_idx]
            move()
            if (self.state == self.solved_cube).all():
                done = 1
                reward = 1
            else:     
                done = 0
                reward = -.1
            return reward, self.state, done

    def reset(self):
        self.state = self.get_solved_cube()
        
    def get_solved_cube(self):
        arr = np.arange(54)//9
        arr = arr.reshape(6,3,3)
        return arr
        
    def top(self):
        temp_state = copy.deepcopy(self.state)
        self.state[0,0]=temp_state[1,0]
        self.state[1,0]=temp_state[2,0]
        self.state[2,0]=temp_state[3,0]
        self.state[3,0]=temp_state[0,0]
        self.state[4]=np.rot90(temp_state[4],-1)
        return self.state

    def top_rev(self):
        temp_state = copy.deepcopy(self.state)
        self.state[0,0]=temp_state[3,0]
        self.state[1,0]=temp_state[0,0]
        self.state[2,0]=temp_state[1,0]
        self.state[3,0]=temp_state[2,0]
        self.state[4]=np.rot90(temp_state[4],1)
        return self.state

    def right(self):
        temp_state = copy.deepcopy(self.state)
        self.state[0,:,2]=temp_state[5,:,2]
        self.state[4,:,2]=temp_state[0,:,2]
        self.state[2,:,0]=np.flip(temp_state[4,:,2],0)
        self.state[5,:,2]=np.flip(temp_state[2,:,0],0)
        self.state[1]=np.rot90(temp_state[1],-1)
        return self.state

    def right_rev(self):
        temp_state = copy.deepcopy(self.state)
        self.state[0,:,2]=temp_state[4,:,2]
        self.state[4,:,2]=np.flip(temp_state[2,:,0],0)
        self.state[2,:,0]=np.flip(temp_state[5,:,2],0)
        self.state[5,:,2]=temp_state[0,:,2]
        self.state[1]=np.rot90(temp_state[1],1)
        return self.state
        
    def left(self):
        temp_state = copy.deepcopy(self.state)
        self.state[0,:,0]=temp_state[4,:,0]
        self.state[4,:,0]=np.flip(temp_state[2,:,2],0)
        self.state[2,:,2]=np.flip(temp_state[5,:,0],0)
        self.state[5,:,0]=temp_state[0,:,0]
        self.state[3]=np.rot90(temp_state[3],-1)
        return self.state

    def left_rev(self):
        temp_state = copy.deepcopy(self.state)
        self.state[0,:,0]=temp_state[5,:,0]
        self.state[4,:,0]=temp_state[0,:,0]
        self.state[2,:,2]=np.flip(temp_state[4,:,0],0)
        self.state[5,:,0]=np.flip(temp_state[2,:,2],0)
        self.state[3]=np.rot90(temp_state[3],1)
        return self.state
        
    def bottom(self):
        temp_state = copy.deepcopy(self.state)
        self.state[0,2,:]=temp_state[3,2,:]
        self.state[1,2,:]=temp_state[0,2,:]
        self.state[2,2,:]=temp_state[1,2,:]
        self.state[3,2,:]=temp_state[2,2,:]
        self.state[5]=np.rot90(temp_state[5],-1)
        return self.state

    def bottom_rev(self):
        temp_state = copy.deepcopy(self.state)
        self.state[0,2]=temp_state[1,2]
        self.state[1,2]=temp_state[2,2]
        self.state[2,2]=temp_state[3,2]
        self.state[3,2]=temp_state[0,2]
        self.state[5]=np.rot90(temp_state[5],1)
        return self.state

    def back(self):
        temp_state = copy.deepcopy(self.state)
        self.state[1,:,2]=np.flip(temp_state[5,2,:],0)
        self.state[5,2,:]=temp_state[3,:,0]
        self.state[3,:,0]=np.flip(temp_state[4,0,:],0)
        self.state[4,0,:]=temp_state[1,:,2]
        self.state[2]=np.rot90(temp_state[2],-1)
        return self.state

    def back_rev(self):
        temp_state = copy.deepcopy(self.state)
        self.state[1,:,2]=temp_state[4,0,:]
        self.state[5,2,:]=np.flip(temp_state[1,:,2],0)
        self.state[3,:,0]=temp_state[5,2,:]
        self.state[4,0,:]=np.flip(temp_state[3,:,0],0)
        self.state[2]=np.rot90(temp_state[2],1)
        return self.state
    
    def front(self):
        temp_state = copy.deepcopy(self.state)
        self.state[1,:,0]=temp_state[4,2,:]
        self.state[5,0,:]=np.flip(temp_state[1,:,0],0)
        self.state[3,:,2]=temp_state[5,0,:]
        self.state[4,2,:]=np.flip(temp_state[3,:,2],0)
        self.state[0]=np.rot90(temp_state[0],-1)
        return self.state

    def front_rev(self):
        temp_state = copy.deepcopy(self.state)
        self.state[1,:,0]=np.flip(temp_state[5,0,:],0)
        self.state[5,0,:]=temp_state[3,:,2]
        self.state[3,:,2]=np.flip(temp_state[4,2,:],0)
        self.state[4,2,:]=temp_state[1,:,0]
        self.state[0]=np.rot90(temp_state[0],1)
        return self.state

if __name__ == '__main__':
    cube = Cube()
    cube.right()
    print(cube.state)