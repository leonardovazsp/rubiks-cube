import requests
import json
import pickle
import numpy as np
import os
from PIL import Image

print('Testing API...')

# url = 'http://rubiks.ngrok.io'
url = 'http://192.168.1.111:5000'
add_url = url + '/add'
train_url = url + '/train'
predict_url = url + '/predict'

for i in range(5):

    state = np.random.randint(0, 6, (6, 3, 3), dtype=np.uint8)

    img1 = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    images = [img1, img2]

    payload = pickle.dumps({'images': images, 'state': state})

    response = requests.post(add_url, data=payload)
    print(response.json())