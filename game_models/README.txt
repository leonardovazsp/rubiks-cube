1. Create the cube:
from local_cube import * 
cube = Cube() #

2. cube.status -> <class 'numpy.ndarray'> / shape: (6 faces, 3 rows, 3 squares) = 64 squares in 6 colors

3. Flatten a numpy.ndarray: cube.status.flatten()
tip:
- ravel() will affect the original array, 
- while changes to the array returned by flatten() will not affect the original array.