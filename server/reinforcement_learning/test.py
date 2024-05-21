from environment import Cube
from agent import Agent
import copy

cube = Cube()
agent = Agent(cube, name='Model4_2l_Double')
agent.load_models()

n_scramble = 12
max_steps = 100
prediction_solved = list()
for k in range(1000):
    cube.reset()
    cube.scramble(n_scramble)

    step_counter = 0
    while step_counter < n_scramble:
        state = copy.deepcopy(cube.state)
        action_pred = agent.get_action(state, 0.001)
        cube.step(action_pred)

        is_solved = cube.is_solved()
        step_counter += 1

        if is_solved:
            break

    prediction_solved.append(is_solved)

print(f"% Cubic Solved: {sum(prediction_solved) / len(prediction_solved) * 100}")



