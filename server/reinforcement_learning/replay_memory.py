import random
from collections import namedtuple, deque
# import torch

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'state_', 'done'))

class ReplayMemory(object):
    """ https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) #without replacement

    def __len__(self):
        return len(self.memory)
    
if __name__ == '__main__':
    import numpy as np
    replay = ReplayMemory(5)

    for _ in range(10):
        replay.push(np.random.randint(0, 6, (6, 3, 3)), 
                    np.random.randint(0, 12),
                    np.random.randint(0, 6, (6, 3, 3)),
                    np.random.choice([-1, 100], p=[0.9, 0.1]))
    
    batch = Experience(*zip(*replay.memory))
    print(batch.action)
    print(batch.reward)



