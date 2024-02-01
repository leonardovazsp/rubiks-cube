from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
import json
from dataset import Dataset
import numpy as np

dataset = Dataset()

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
    dataset.save_example(images, state)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)