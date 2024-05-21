import numpy as np
import copy
import random

colors = {
    0: "\033[91m",          # Red
    1: "\033[94m",          # Blue
    2: "\033[38;5;208m",    # Orange
    3: "\033[92m",          # Green
    4: "\033[97m",          # White
    5: "\033[93m",          # Yellow
    "reset": "\033[0m"  # Reset color
}

class Cube():
    def __init__(self, state=None):
        if state is None:
            self.state = self.get_solved_cube()
        else:
            self.state = state
        self.solved_cube = self.get_solved_cube()
        self.moves = [self.right, self.right_rev, self.left, self.left_rev, self.top, self.top_rev, self.bottom, self.bottom_rev, self.front, self.front_rev, self.back, self.back_rev]
        self.last_move = None
        self.double_move = False
        self.reward = 100.

    def __repr__(self):
        return str(self.state)
    
    def __str__(self):
        self.print_cube()
        return ""
    
    def set_reward(self, reward):
        self.reward = reward
    
    def step(self, move_idx):
            self.moves[move_idx]()
            if (self.state == self.solved_cube).all():
                reward = self.reward
            else:
                reward = 0
            return reward, self.state
    
    def get_random_step(self):
        moves = list(range(12))
        if self.last_move:
            if self.double_move:
                moves.remove(self.last_move)

            if self.last_move % 2 == 0:
                moves.remove(self.last_move+1)

            else:
                moves.remove(self.last_move-1)

        move_idx = random.choice(moves)
        if move_idx == self.last_move:
            self.double_move = True
        else:
            self.double_move = False

        self.last_move = move_idx
        return move_idx
    
    def get_next_state(self, move_idx):
        temp_state = copy.deepcopy(self.state)
        self.moves[move_idx]()
        if (self.state == self.solved_cube).all():
            reward = self.reward
        else:
            reward = 0
        return_state = copy.deepcopy(self.state)
        self.state = temp_state
        return return_state, reward

    def reset(self):
        self.state = self.get_solved_cube()
        return self.state
    
    def scramble(self, n=20):
        self.reset()
        moves = list(range(12))
        for i in range(n):
            move_idx = random.choice(moves)
            self.step(move_idx)
            moves = list(range(12))
            if move_idx%2 == 0:
                moves.remove(move_idx+1)
            else:
                moves.remove(move_idx-1)

    def is_solved(self):
        return (self.state == self.solved_cube).all()

    def shuffle(self, n=20):
        self.reset()
        for i in range(n):
            move = random.choice(self.moves)
            move()
        if (self.state == self.solved_cube).all():
            self.shuffle(1)
        return self.state
        
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
    
    def print_cube(self):
   
        # Print Upper part
        for i in range(3):
            print("                ", end="")
            for j in range(3):
                print(f"{colors[self.state[4, i, j]]}██{colors['reset']}", end="")
            print()
        print()
        
        # Print Middle part
        for i in range(3):
            for k in range(4):
                for j in range(3):
                    print(f"{colors[self.state[k, i, j]]}██{colors['reset']}", end="")
                print("  ", end="")
            print()
        print()
        
        # Print Lower part
        for i in range(3):
            print("                ", end="")
            for j in range(3):
                print(f"{colors[self.state[5, i, j]]}██{colors['reset']}", end="")
            print()

if __name__ == '__main__':
    cube = Cube()
    # cube.scramble(1)
    # cube.step(1)
    state = copy.deepcopy(cube.state)
    action = cube.get_random_step()
    reward, state_ = cube.step(action)
    state_ = copy.deepcopy(state_)
    dones = int(cube.is_solved())

    print(state)
    print(state_)
    print(reward)
    print(dones)