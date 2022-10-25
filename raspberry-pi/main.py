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

# Allow the camera to warmup
time.sleep(0.1)

# Define the type of server
server_type = config['server_type']
if server_type == 'master':
    # Master server is responsible for rotating the cube as well as getting the images from the camera
    from cube import Cube
    cube = Cube()
    camera_server_url = config['camera_server_url']
    master = True
elif server_type == 'camera':
    # Camera server is responsible for getting the images from the camera
    master = False

app = Flask(__name__)

@app.route('/', methods=['POST'])
def receive_request():
    # Receive a request from the client with the move to be made on the cube as post data
    # Rotate the cube according to the move
    # Get image from the local camera and from the camera server
    # Get the cube state
    # Return images and cube state as json
    if master:
        move = request.data.decode('utf-8')
        if move == 'reset':
            cube.reset()
        elif move in cube.moves_list:
            move = getattr(cube, move)()
            move
        elif move == 'random':
            cube.random_move()
        cube_state = cube.get_cube_state()
        # Get image from the local camera
        camera.capture(rawCapture, format='bgr')
        image = rawCapture.array
        rawCapture.truncate(0)
        # Get image from the camera server
        r = requests.post(camera_server_url, data='get_img')
        image_server = pickle.loads(r.content)
        response = pickle.dumps([image, image_server, cube_state])
        return response
    else:
        # Get image from the local camera
        data = request.data.decode('utf-8')
        if not data == 'get_img':
            return None
        camera.capture(rawCapture, format='bgr')
        image = rawCapture.array
        rawCapture.truncate(0)
        # Return data as pickle
        return pickle.dumps(image)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)