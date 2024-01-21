'''
API to interact with the Rubik's Cube and get images from the camera as well as the cube state.
'''
import os
import time
import json
import requests
import pickle
import hmac
import hashlib
import subprocess
import ngrok
from picamera import PiCamera
from picamera.array import PiRGBArray
from flask import Flask, jsonify, request

port = os.environ["RUBIKS_PORT"]
domain = os.environ["NGROK_DOMAIN"]
listener = ngrok.forward(port, domain=domain, authtoken_from_env=True)
print(f"Ingress established at {listener.url()}")

WEBHOOK_SECRET = b'leo123456'

time.sleep(5)
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
server_type = os.environ["SERVER_TYPE"]
master = False
moving = False

if server_type == 'master':
    # Master server is responsible for rotating the cube as well as getting the images from the camera
    from cube import Cube
    cube = Cube()
    camera_server_url = 'http://' + find_ip('camera') + ':8000/'
    master = True

app = Flask(__name__)

@app.route('/get_cube_state/', methods=['GET'])
def get_cube_state():
    """
    Get the state of the cube.
    """

    if moving:
        return 400
    
    if master:
        return jsonify(cube.get_cube_state())
    return jsonify(cube.get_cube_state())

@app.route('/get_images/', methods=['GET'])
def get_images():
    """
    Get images from the local camera and from the remote camera.
    Return images as pickle.
    """

    if moving:
        return 400
    
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
        moving = True
        move = request.data.decode('utf-8')
        if move == 'reset':
            cube.reset()

        elif move in cube.moves_list:
            move = getattr(cube, move)()
            move

        elif move == 'random':
            cube.random_move()

        time.sleep(0.2)
        moving = False
        return 200
    
@app.route('/update', methods=['POST'])
def update():
    """
    Update the script when a new push is made to the repo.
    """

    payload = request.data
    signature = request.headers.get('X-Hub-Signature')
    if not is_valid_signature(payload, signature):
        return 'Invalid signature', 400
    
    subprocess.Popen(['./update_script.sh'])
    
def is_valid_signature(payload, signature):
    if not signature:
        return False
    
    sha_name, signature = signature.split('=')
    if sha_name != 'sha1':
        return False
    
    mac = hmac.new(WEBHOOK_SECRET, msg=payload, digestmod=hashlib.sha1)
    return hmac.compare_digest(mac.hexdigest(), signature)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)