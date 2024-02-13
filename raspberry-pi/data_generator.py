from camera import Cameras
import requests
import pickle

cameras = Cameras()

url = 'http://rubiks.ngrok.io'

class Generator():
    def __init__(self, cube):
        self.cube = cube

    def generate(self, examples=100, scramble=1):
        for i in range(examples):
            self.cube.scramble(scramble)
            img0, img1 = cameras.capture()
            state = self.cube.state
            payload = {'images': [img0, img1], 'state': state}
            payload = pickle.dumps(payload)
            sent = False
            while not sent:
                try:
                    response = requests.post(url + '/add', data=payload)
                    if response.status_code == 200:
                        sent = True

                except:
                    print('Failed to send request')
                    continue

if __name__ == '__main__':
    from cube import Cube
    cube = Cube()
    cube.driver.activate()
    cube.driver.set_delay(1000)
    gen = Generator(cube)