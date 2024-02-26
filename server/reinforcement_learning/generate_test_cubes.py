from environment import Cube
import copy
import pickle
from fire import Fire

def get_scrambled_cubes(n_cubes, n_scrambles):
        cubes = []
        for i in range(n_cubes):
            cube = Cube()
            cube.scramble(n_scrambles)
            cubes.append(copy.deepcopy(cube.state))
        return cubes

def main(total_levels=20,
         n_cubes_per_level=10,
         save_path='test_cubes.pkl'):
    test_cubes = {}
    for i in range(1, total_levels+1):
        test_cubes[i] = get_scrambled_cubes(n_cubes_per_level, i)

    with open('test_cubes.pkl', 'wb') as f:
        pickle.dump(test_cubes, f)

    print(f"{n_cubes_per_level} cubes per level from 1 to {total_levels} saved to {save_path}")

if __name__ == '__main__':
     Fire(main)