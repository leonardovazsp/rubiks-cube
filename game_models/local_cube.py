import time
import numpy as np
import copy
import pickle
import inspect
import random

class Cube():
    def __init__(self, status=None):
        if status == None:
            self.status = self.get_solved_cube()
        else:
            self.status = status
        self.solved_cube = self.get_solved_cube()
        self.moves_list = ['right', 'right_rev', 'left', 'left_rev', 'top', 'top_rev', 'bottom', 'bottom_rev', 'front', 'front_rev', 'back', 'back_rev']
        self.moves = [self.right, self.right_rev, self.left, self.left_rev, self.top, self.top_rev, self.bottom, self.bottom_rev, self.front, self.front_rev, self.back, self.back_rev]
        
    def step(self, move_idx):
            move = self.moves[move_idx]
            move()
            if (self.status == self.solved_cube).all():
                done = 1
                reward = 1
            else:     
                done = 0
                reward = -.1
            return reward, self.status, done

    def reset(self):
        self.status = self.get_solved_cube()
        
    def get_solved_cube(self):
        arr = np.arange(54)//9
        arr = arr.reshape(6,3,3)
        return arr
        
    def top(self):
        temp_status = copy.deepcopy(self.status)
        self.status[0,0]=temp_status[1,0]
        self.status[1,0]=temp_status[2,0]
        self.status[2,0]=temp_status[3,0]
        self.status[3,0]=temp_status[0,0]
        self.status[4]=np.rot90(temp_status[4],-1)
        return self.status

    def top_rev(self):
        temp_status = copy.deepcopy(self.status)
        self.status[0,0]=temp_status[3,0]
        self.status[1,0]=temp_status[0,0]
        self.status[2,0]=temp_status[1,0]
        self.status[3,0]=temp_status[2,0]
        self.status[4]=np.rot90(temp_status[4],1)
        return self.status

    def right(self):
        temp_status = copy.deepcopy(self.status)
        self.status[0,:,2]=temp_status[5,:,2]
        self.status[4,:,2]=temp_status[0,:,2]
        self.status[2,:,0]=np.flip(temp_status[4,:,2],0)
        self.status[5,:,2]=np.flip(temp_status[2,:,0],0)
        self.status[1]=np.rot90(temp_status[1],-1)
        return self.status

    def right_rev(self):
        temp_status = copy.deepcopy(self.status)
        self.status[0,:,2]=temp_status[4,:,2]
        self.status[4,:,2]=np.flip(temp_status[2,:,0],0)
        self.status[2,:,0]=np.flip(temp_status[5,:,2],0)
        self.status[5,:,2]=temp_status[0,:,2]
        self.status[1]=np.rot90(temp_status[1],1)
        return self.status
        
    def left(self):
        temp_status = copy.deepcopy(self.status)
        self.status[0,:,0]=temp_status[4,:,0]
        self.status[4,:,0]=np.flip(temp_status[2,:,2],0)
        self.status[2,:,2]=np.flip(temp_status[5,:,0],0)
        self.status[5,:,0]=temp_status[0,:,0]
        self.status[3]=np.rot90(temp_status[3],-1)
        return self.status

    def left_rev(self):
        temp_status = copy.deepcopy(self.status)
        self.status[0,:,0]=temp_status[5,:,0]
        self.status[4,:,0]=temp_status[0,:,0]
        self.status[2,:,2]=np.flip(temp_status[4,:,0],0)
        self.status[5,:,0]=np.flip(temp_status[2,:,2],0)
        self.status[3]=np.rot90(temp_status[3],1)
        return self.status
        
    def bottom(self):
        temp_status = copy.deepcopy(self.status)
        self.status[0,2,:]=temp_status[3,2,:]
        self.status[1,2,:]=temp_status[0,2,:]
        self.status[2,2,:]=temp_status[1,2,:]
        self.status[3,2,:]=temp_status[2,2,:]
        self.status[5]=np.rot90(temp_status[5],-1)
        return self.status

    def bottom_rev(self):
        temp_status = copy.deepcopy(self.status)
        self.status[0,2]=temp_status[1,2]
        self.status[1,2]=temp_status[2,2]
        self.status[2,2]=temp_status[3,2]
        self.status[3,2]=temp_status[0,2]
        self.status[5]=np.rot90(temp_status[5],1)
        return self.status

    def back(self):
        temp_status = copy.deepcopy(self.status)
        self.status[1,:,2]=np.flip(temp_status[5,2,:],0)
        self.status[5,2,:]=temp_status[3,:,0]
        self.status[3,:,0]=np.flip(temp_status[4,0,:],0)
        self.status[4,0,:]=temp_status[1,:,2]
        self.status[2]=np.rot90(temp_status[2],-1)
        return self.status

    def back_rev(self):
        temp_status = copy.deepcopy(self.status)
        self.status[1,:,2]=temp_status[4,0,:]
        self.status[5,2,:]=np.flip(temp_status[1,:,2],0)
        self.status[3,:,0]=temp_status[5,2,:]
        self.status[4,0,:]=np.flip(temp_status[3,:,0],0)
        self.status[2]=np.rot90(temp_status[2],1)
        return self.status
    
    def front(self):
        temp_status = copy.deepcopy(self.status)
        self.status[1,:,0]=temp_status[4,2,:]
        self.status[5,0,:]=np.flip(temp_status[1,:,0],0)
        self.status[3,:,2]=temp_status[5,0,:]
        self.status[4,2,:]=np.flip(temp_status[3,:,2],0)
        self.status[0]=np.rot90(temp_status[0],-1)
        return self.status

    def front_rev(self):
        temp_status = copy.deepcopy(self.status)
        self.status[1,:,0]=np.flip(temp_status[5,0,:],0)
        self.status[5,0,:]=temp_status[3,:,2]
        self.status[3,:,2]=np.flip(temp_status[4,2,:],0)
        self.status[4,2,:]=temp_status[1,:,0]
        self.status[0]=np.rot90(temp_status[0],1)
        return self.status

if __name__ == '__main__':
    cube = Cube()
    print(cube.status)