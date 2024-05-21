from replay_memory import ReplayMemory, Experience
from model import DQN

import numpy as np
import torch
import copy


class Agent():
    def __init__(self, env, gamma=0.9,
                 epsilon=0.9, lr=0.001, 
                 mem_size=1000000, batch_size=64, 
                 eps_dec=5e-7, eps_min=0.01, 
                 replace=1000, name='model1.pt',
                 checkpoint_dir='temp/') -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.n_actions = len(env.moves)
        self.input_shape = self.env.state.shape
        self.replace_target_count = replace
        self.step_counter = 0

        self.memory = ReplayMemory(mem_size)
        self.q_eval = DQN(self.input_shape,
                          self.n_actions,
                          self.lr,
                          self.name + "_eval.pt",
                          self.checkpoint_dir)
        self.q_next = DQN(self.input_shape,
                    self.n_actions,
                    self.lr,
                    self.name + "_next.pt",
                    self.checkpoint_dir)        

    def get_action(self, state, epsilon=None):
        if epsilon is None:
            specified_epsilon = self.epsilon
        else: 
            specified_epsilon = epsilon

        if np.random.rand() < specified_epsilon:
            action = self.env.get_random_step()
        else:
            state = torch.tensor(state, dtype=torch.float).reshape(1, -1) #play needs update
            actions = self.q_eval(state)
            action = torch.argmax(actions).item()
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.push(state, action, reward, state_, done)

    def sample_memory(self):
        sample = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*sample))
        
        actions = torch.tensor(batch.action)
        states = torch.tensor(np.array(batch.state), dtype=torch.float)
        rewards = torch.tensor(batch.reward)
        states_ = torch.tensor(np.array(batch.state_), dtype=torch.float)
        dones = torch.tensor(batch.done)
        return actions, states, rewards, states_, dones

    def replace_target_network(self):
        if self.step_counter % self.replace_target_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        # self.q_eval.load_checkpoint(q_eval_file)
        # self.q_next.load_checkpoint(q_next_file)
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.q_eval._optimizer.zero_grad()
        self.replace_target_network()

        actions, states, rewards, states_, dones =  self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval(states)[indices, actions]
        q_next = self.q_next(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        # if dones.any() == 1:
        #     # print(q_next, dones)
        #     print(dones, rewards)

        q_target = rewards + self.gamma * q_next
        loss = self.q_eval.loss(q_target, q_pred)

        loss.backward()
        self.q_eval._optimizer.step()
        
        self.step_counter += 1
        self.decrement_epsilon()

        return loss
    
    def learn_double(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.q_eval._optimizer.zero_grad()
        self.replace_target_network()

        actions, states, rewards, states_, dones =  self.sample_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval(states)[indices, actions]
        q_next = self.q_next(states_)
        q_eval = self.q_eval(states_)

        max_action = torch.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next[indices, max_action]
        loss = self.q_eval.loss(q_target, q_pred)

        loss.backward()
        self.q_eval._optimizer.step()
        
        self.step_counter += 1
        self.decrement_epsilon()

        # print('q_pred', q_pred)
        # print('q_eval', q_eval)
        # print(max_action)
        # print(q_next[indices, max_action])
        return loss



if __name__ == '__main__':
    from environment import Cube

    cube = Cube()
    agent = Agent(cube, batch_size=64, eps_dec=1e-4, name='testing')

    for i in range(10000):
        cube.scramble(1)
        state = copy.deepcopy(cube.state)
        action = agent.get_action(state)
        reward, state_ = cube.step(action)
        state_ = copy.deepcopy(state_)
        done = cube.is_solved()

        agent.store_transition(state, action, reward, state_, done)
        loss = agent.learn_double()

        cube.reset()

        if i % 1000 == 0:
            print(f"Iter: {i}, Loss: {loss}, Epsilon: {agent.epsilon}")

    for move in range(12):
        cube.reset()
        cube.step(move)
        state = copy.deepcopy(cube.state)

        print(f"Move: {move}, Action: {agent.get_action(state, 0.001)}")
    
    # agent.save_models()
    



