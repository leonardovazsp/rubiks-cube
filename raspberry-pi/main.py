'''
API to interact with the Rubik's Cube and get images from the camera as well as the cube state.
'''

from flask import Flask, jsonify, request
import numpy as np
import os
import random
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import json
import requests
import pickle

# Load config
with open('config.json') as f:
    config = json.load(f)

# Initialize camera
resolution = config['camera']['resolution']
camera = PiCamera()
camera.resolution = resolution
rawCapture = PiRGBArray(camera, size=resolution)

def find_ip(server_type):
    '''
    Scan through the network to find the server.
    '''

    ip = None
    for i in range(100, 256):
        ip = '192.168.1.' + str(i)
        try:
            r = requests.get('http://' + ip + ':8000/get_device/', timeout=0.2)
            if r.text == server_type:
                print('Found ' + server_type + ' server at ' + ip)
                return ip
        except:
            pass
    raise Exception('Could not find ' + server_type + ' server')
    
# Allow the camera to warmup
time.sleep(0.1)

# Define the type of server
server_type = config['server_type']
master = False

if server_type == 'master':
    # Master server is responsible for rotating the cube as well as getting the images from the camera
    from cube import Cube
    cube = Cube()
    camera_server_url = 'http://' + find_ip('camera') + ':8000/'
    master = True

app = Flask(__name__)

@app.route('/get_cube_state/', methods=['GET'])
def get_cube_state():
    return jsonify(cube.get_cube_state())

@app.route('/get_images/', methods=['GET'])
def get_images():
    """
    Get images from the local camera and from the remote camera.
    Return images as pickle.
    """
    print("")
    if master:
        # Get image from the local camera
        print("Getting image from local camera (server)")
        camera.capture(rawCapture, format='bgr')
        image = rawCapture.array
        rawCapture.truncate(0)

        # Get image from the camera server
        print("Getting image from the camera server")
        r = requests.get(camera_server_url + 'get_images/')
        image_server = pickle.loads(r.content, encoding='bytes')
        response = pickle.dumps([image, image_server])
        return response

    # Get image from the local camera (this will run only on the remote camera)
    print("Getting image from local camera")
    camera.capture(rawCapture, format='bgr')
    image = rawCapture.array
    rawCapture.truncate(0)
    return pickle.dumps(image)

@app.route('/get_device/', methods=['GET'])
def get_device():
    return server_type

@app.route('/move', methods=['POST'])
def move():
    """
    Receive a request from the client with the move to be made and rotate the
    cube according to the move.

    """
    if master:
        move = request.data.decode('utf-8')
        if move == 'reset':
            cube.reset()

        elif move in cube.moves_list:
            move = getattr(cube, move)()
            move

        elif move == 'random':
            cube.random_move()

        return 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)