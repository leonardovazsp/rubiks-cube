from environment import Cube
from agent import Agent
import numpy as np
import copy

from collections import Counter

ngames = 100000
nscramble = 4

cube = Cube()
agent = Agent(cube, batch_size=128, eps_dec=1e-5, name='Model4_3l_Double')

ts = list()
for game in range(ngames):
    cube.reset()
    cube.scramble(nscramble)

#     # for varying scrambles
#     scramble = np.random.choice(np.arange(nscramble)) #p=[0.5, 0.3, 0.2]
#     cube.scramble(scramble)
# #     ts.append(scramble)
# # print(Counter(ts).keys())
# # print(Counter(ts).values())

    step_counter = 0
    while step_counter < nscramble * 2:
        state = copy.deepcopy(cube.state)
        action = agent.get_action(state)
        reward, state_ = cube.step(action)
        state_ = copy.deepcopy(state_) 
        done = cube.is_solved()

        agent.store_transition(state, action, reward, state_, done)
        step_counter += 1

        if done:
            break
    
    loss = agent.learn_double()

    if game % 1000 == 0:
        print(f"Iter: {game}, Loss: {loss}, Epsilon: {agent.epsilon}")

agent.save_models()