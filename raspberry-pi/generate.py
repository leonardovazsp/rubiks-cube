from generator import Generator
from cube import Cube

cube = Cube()
cube.driver.activate()
cube.driver.set_delay(2500)
gen = Generator(cube, 'data')
