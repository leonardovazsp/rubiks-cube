import requests
import json
import pickle
import numpy as np
import os
from PIL import Image
import subprocess
import time

print('Testing API...')

url = 'http://rubiks.ngrok.io'
# url = 'http://192.168.7.208:5000'
# url = "http://40.118.184.142:5000"
add_url = url + '/add'
train_url = url + '/train'
predict_url = url + '/predict'

for i in range(20):

    state = np.random.randint(0, 6, (6, 3, 3), dtype=np.uint8)

    start = time.time()
    subprocess.check_call(['rpicam-jpeg', '-o', '0.jpeg', '-t', '2', '--width', '244', '--height', '244', '-v', '0', '--camera', '0'])
    subprocess.check_call(['rpicam-jpeg', '-o', '1.jpeg', '-t', '2', '--width', '244', '--height', '244', '-v', '0', '--camera', '1'])
    
    img1 = Image.open('0.jpeg')
    img2 = Image.open('1.jpeg')

    end = time.time()

    print(f"Time taken: {(end - start):2f}")
    # img1 = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    # img2 = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    images = [img1, img2]

    payload = pickle.dumps({'images': images, 'state': state})
    print('Sending request...')
    sent = False
    while not sent:
        try:
            response = requests.post(add_url, data=payload)
            print(response.json())
            sent = True
        except:
            print('Failed to send request')
            print('Trying again...')
            continue

    print(response.json())