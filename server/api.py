from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
import json



app = Flask(__name__)

@app.route('/add', methods=['POST'])
def add():
    """
    Receives 2 images (jpg) and cube state (npy) and saves to data folder.
    Data is received in pickle format.
    """
    data = pickle.loads(request.data)
    images = data['images']
    state = data['state']
    example_num = get_example_num()

    for i, image in enumerate(images):
        with open(f'data/{example_num}_{i}.jpg', 'wb') as f:
            f.write(image)

    with open(f'data/{example_num}.npy', 'wb') as f:
        np.save(f, state)

    return jsonify({'success': True})
    
@app.route('/train', methods=['POST'])
def train():
    data = json.loads(request.data)
    epochs = data.get('epochs', 10)
    lr = data.get('lr', 0.001)
    batch_size = data.get('batch_size', 32)
    shuffle = data.get('shuffle', True)
    save_model = data.get('save_model', True)
    save_history = data.get('save_history', True)
    split = data.get('split', 0.2)
    pass

@app.route('/predict', methods=['POST'])
def predict():
    pass