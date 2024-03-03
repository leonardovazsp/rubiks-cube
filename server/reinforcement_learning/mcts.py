from environment import Cube
import numpy as np
import torch
from collections import deque
import pprint
from tqdm import tqdm

class Node():
    def __init__(self, state, action=None):
        self.state = state
        self.children = []
        self.parent = None
        self.action = action
        self.memory = {}

    def __repr__(self):
        memory = {action: {k: round(v, 4) for k, v in params.items()} for action, params in self.memory.items()}
        return pprint.pformat(memory)

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def is_leaf(self):
        return len(self.children) == 0
    
    def get_best_action(self, c):
        max_value = float('-inf')
        best_action = None
        total_visits = sum([params['N'] for action, params in self.memory.items()])
        for action, params in self.memory.items():
            N, W, L, P = params.values()
            U = c * P * np.sqrt(total_visits + 1) / (1 + N)
            Q = W - L
            value = Q + U
            if value > max_value:
                max_value = value
                best_action = action
        return best_action
    

class MCTS():
    def __init__(self, root_state, model, device):
        self.solved_state = Cube().state
        self.root = Node(root_state)
        self.model = model
        self.device = device
        self._init_root()

    def _get_policy_value(self, state):
        self.model.eval()
        state = torch.tensor(state).long().view(-1, 54).to(self.device)
        with torch.no_grad():
            policy, value = self.model(state)
        policy = policy.squeeze().detach().cpu().numpy()
        value = value.squeeze().detach().cpu().numpy()
        return policy, value
    
    def _init_memory(self, node, policy):
        for action in range(12):
            node.memory[action] = {'N': 0, 'W': -0.2, 'L': 0, 'P': policy[action]}

    def _init_root(self):
        policy, value = self._get_policy_value(self.root.state)
        self._init_memory(self.root, policy)

    def is_solved(self, state):
        return (state == self.solved_state).all()
    
    def _select(self, node, c):
        while not node.is_leaf():
            action = node.get_best_action(c)
            node = node.children[action]
        return node
    
    def _expand(self, node):
        cube = Cube(node.state)
        for action in range(12):
            next_state, _ = cube.get_next_state(action)
            node.add_child(Node(next_state, action))

    def _backpropagate(self, node, value, v):
        while node:
            action_taken = node.action
            if action_taken is not None:
                node.parent.memory[action_taken]['N'] += 1
                node.parent.memory[action_taken]['W'] = max(node.parent.memory[action_taken]['W'], value)
                node.parent.memory[action_taken]['L'] += v
            node = node.parent

    def shortest_path(self):
        visited = set()
        queue = deque([(self.root, [])])

        if self.is_solved(self.root.state):
            return []
        
        while queue:
            node, path = queue.popleft()
            
            state_tuple = (tuple(node.state.flatten()), tuple(path))
            if state_tuple in visited:
                continue

            visited.add(state_tuple)

            for child in node.children:
                if self.is_solved(child.state):
                    return path + [child.action]
                queue.append((child, path + [child.action]))

        return []
        
    def run(self, c, v, max_iterations=1000):
        done = False
        for i in range(max_iterations):
            print(f"Iteration {i}", end='\r')
            leaf_node = self._select(self.root, c)
            self._expand(leaf_node)
            children_states = [child.state for child in leaf_node.children]
            states = [leaf_node.state] + children_states
            states = np.array(states)
            policy, value = self._get_policy_value(states)
            policy = policy[1:]
            value = value[0]
            for j, child in enumerate(leaf_node.children):
                self._init_memory(child, policy[j])

                if self.is_solved(child.state):
                    print(f"Solved in {i} iterations!")
                    done = True
                    break

            self._backpropagate(leaf_node, value, v)

            if done:
                break

        path = self.shortest_path()
        return path
                
            
