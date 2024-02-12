from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
import json
from dataset import Dataset
import numpy as np
import sys
from trainer import Trainer
import model as models

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
    model_name = data.get('model', 'ColorRecognizer')
    checkpoint = data.get('checkpoint', None)
    epochs = data.get('epochs', 10)
    lr = data.get('lr', 0.001)
    batch_size = data.get('batch_size', 32)
    shuffle = data.get('shuffle', True)
    save_model = data.get('save_model', True)
    save_history = data.get('save_history', True)
    split = data.get('split', 0.2)
    optimizer = data.get('optimizer', 'Adam')
    criterion = data.get('criterion', 'CrossEntropyLoss')
    device = data.get('device', 'cpu')
    wandb_project = data.get('wandb_project', None)
    save_dir = data.get('save_dir', 'models')
    kwargs = data.get('kwargs', {})
    model = models.__dict__[model_name]()
    trainer = Trainer(model, dataset, optimizer, criterion, device, batch_size, shuffle, split, lr, wandb_project, save_dir, **kwargs)
    trainer.train(epochs)

@app.route('/predict', methods=['POST'])
def predict():
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)