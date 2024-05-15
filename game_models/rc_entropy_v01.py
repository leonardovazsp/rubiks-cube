from game_models.local_cube import * # When from the notebook
#from local_cube import * # When from this file
import random

class RC_entropy():
    def __init__(self, max_number_scrambles=50, number_moves_allowed=10):
        """
        max_number_scrambles: max number of scrambles that can be performed.
        number_moves_allowed: number of moves the agent can perform before truncate the game.
        """
        print("Version: 13/05/2024 - 13:06")
        # Initialize the cube
        self.cube = Cube()
        # The solved state
        self.initial_state = self.cube.status.flatten()
        # How many scrambles as max
        self.max_number_scrambles = max_number_scrambles
        # Help to truncate the game
        self.number_moves_allowed = number_moves_allowed
        self.number_of_move=None # this variable helps to know when truncate the game
        # Define space of the environment and actions
        self.action_space = 12
        self.environment_space = 54
        # Define the old_state that shows the state you start the episode with and its entropy
        self.old_state = None # This is gonna be selected in reset method
        self.old_entropy = None # This is gonna be calculated in reset method

    # 1. Methods for the main of the game
    def reset(self, seed=None):
        ''' Start the game'''
        # Reset the old_entropy
        self.old_entropy = 0
        # Reset the number of moves the agent will perform before lose (truncate)
        self.reset_number_moves_count()
        # Scrumble the cube
        if seed: random.seed(seed)
        number_scrambles = self.get_number_moves() # Get a number of scramble
        while self.old_entropy == 0: # Avoid return the completed_state
            # Reset the cube
            self.cube.reset()
            # Reset the best_actions
            self.best_moves = []
            for _ in range(number_scrambles):
                # 12 different possible moves
                move = random.randint(0, 11) # generate a random integer between 0 and 11 (inclusive)
                self.cube.step(move)
                self.best_moves.append(move)
            self.define_old_state_and_entropy()
        self.do_anti_actions_list(self.best_moves)
        #print(f"\n{self.old_entropy=}\n") # Delete
        return  self.old_state, self.anti_action_list[::-1] # Return the environment
    
    def step(self, action:int):
        """ 
        Perform an action
        terminated: When the agent reduces the entropy
        truncated: When the agent finishes its allowed number of moves
        """
        self.cube.step(action)
        new_state = self.cube.status.flatten()
        # Entropy: number of elements different from the solved rubik cube
        new_entropy = len([1 for x, y in zip(self.initial_state, new_state) if x != y])
        if new_entropy == 0: completed = True 
        else: completed=False
        # logic of the game:
        self.number_of_move -= 1 # burn your move
        if new_entropy >= self.old_entropy:
            reward = -1
            terminated = False
            if self.number_of_move <= 0: # When it reached the number allowed of moves
                truncated = True
            else:
                truncated = False
            return new_state, reward, terminated, truncated, completed
        elif self.old_entropy > new_entropy:
            # reward = (R for decrease the entropy) * (R for decrease the most) * (R for use less steps)
            # Game 1:
            #reward = self.number_moves_allowed * (54 - new_entropy) * (self.number_of_move + 1)
            # Game 2:
            if completed:
                reward = self.number_moves_allowed*10
            else:
                reward = self.number_moves_allowed
            terminated = True
            if self.number_of_move <= 0: # When it reached the number allowed of moves
                truncated = True
            else:
                truncated = False
            #print(f"\n{new_entropy=}\n")
            return new_state, reward, terminated, truncated, completed

    # 2. Methods for the the rest of the game
    def get_number_moves(self):
        """Get the number of moves based on the number of scenarios that each level has"""
        scenarios_per_level = [12**(x+1) for x in range(self.max_number_scrambles)] # Number of scenarios per level
        probabilities = [x/sum(scenarios_per_level) for x in scenarios_per_level] # Probability of each level
        got_number_scrambles = np.random.choice(a=len(probabilities), p=probabilities)
        return got_number_scrambles+1 # Select a number of moves = level
    
    def return_prior_state(self, performed_actions):
        for action in performed_actions:
            if action % 2 == 0:
                self.cube.step(action+1)
            else:
                self.cube.step(action-1)

    def do_anti_actions_list(self, actions_list):
        self.anti_action_list = []
        for action in actions_list:
            if action % 2 == 0:
                self.anti_action_list.append(action+1)
            else:
                self.anti_action_list.append(action-1)

    def reset_number_moves_count(self):
        self.number_of_move = self.number_moves_allowed

    def define_old_state_and_entropy(self):
        self.old_state = self.cube.status.flatten()
        self.old_entropy = len([1 for x, y in zip(self.initial_state, self.old_state) if x != y])

    def get_old_entropy(self):
        return self.old_entropy

if __name__ == '__main__':
    env = RC_entropy(max_number_scrambles=10, number_moves_allowed=30)
    state, best_moves = env.reset()
    print(best_moves)
