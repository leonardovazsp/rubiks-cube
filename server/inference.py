from model import RubiksCubeModel
from data_pipeline import SampleGenerator
import json
import torch
import numpy as np
from PIL import Image
import copy
import random
from local_cube import Cube

config = json.load(open('config.json'))
resolution = config['train']['input_shape'][1:]
model = RubiksCubeModel()
model.load_state_dict(torch.load(config['ckpt_path']))

def incorrect(pred, verbose=False):
    pred = np.eye(6)[pred.argmax(axis=-1)]
    pred = pred.reshape(6, 9, 6)
    test = np.sum(pred[:, [1, 3, 5, 7]], axis=(0, 1))
    error_1 = np.sum(np.abs(test - 4))
    test = np.sum(pred[:, [0, 2, 6, 8]], axis=(0, 1))
    error_2 = np.sum(np.abs(test - 4))
    total_error = error_1 + error_2
    if total_error > 0:
        if verbose:
            print(f'{int(error_1)} middle pieces incorrect and {int(error_2)} corner pieces incorrect')
        return True
    if verbose:
        print('Position detected correctly')
    return False

def combine_probailities(preds):
    preds = np.array(preds)
    preds = np.mean(preds, axis=0)
    # preds = torch.softmax(torch.tensor(preds), dim=-1).numpy()
    return preds

def process_preds(pred_1, pred_2):
    pred_2_ = copy.deepcopy(pred_2)
    pred_2[0] = np.rot90(pred_2_[1], 2)
    pred_2[1] = np.rot90(pred_2_[0], 2)
    pred_2[2] = np.rot90(pred_2_[2], -1)
    pred = np.zeros(shape=(6, 3, 3, 6))
    pred[[0, 1, 4]] = pred_1
    pred[[2, 3, 5]] = pred_2
    return pred

def evaluate(pred, label):
    pred = pred.argmax(axis=-1)
    acc = np.sum(pred == label) / 54
    print(f'Accuracy: {acc:.4f}')

class Perceiver():
    def __init__(self):
        self.model = model
        self.resolution = resolution
        self.api = SampleGenerator()
        self.cube = Cube()
        self.moves_list = ['top', 'bottom', 'left', 'right', 'front', 'back', 'back_rev', 'front_rev', 'right_rev', 'left_rev', 'bottom_rev', 'top_rev']
        self.n_samples = 5
        self.lookback = 8
    
    def get_cube_state(self):
        preds = []
        labels = []
        img_1, img_2, label = self.api.get_imgs()
        pred = self.predict(img_1, img_2)
        # pred = torch.softmax(torch.tensor(pred), dim=-1).numpy()
        evaluate(pred, label)
        preds.append(pred)
        labels.append(label)
        moves = iter(self.moves_list * 3)
        while incorrect(pred, verbose=True):
            move = next(moves)
            print(move)
            for i, pred in enumerate(preds):
                cube = Cube(pred)
                # print(f'Pred before move: {np.argmax(pred, axis=-1)}')
                move_ = getattr(cube, move)
                preds[i] = move_()
                # print(f'Pred after move: {np.argmax(preds[i], axis=-1)}')
            self.api.send_request(move = move)
            img_1, img_2, label = self.api.get_imgs()
            pred = self.predict(img_1, img_2)
            print('Eval on new state', end=' ')
            evaluate(pred, label)
            preds.append(pred)
            labels.append(label)
            pred = combine_probailities(preds)
            evaluate(pred, label)
        return pred

    def preprocess_img(self, img):
        img = Image.fromarray(img)
        img = img.resize(resolution)
        img = np.array(img)/255
        light = np.linspace(1, 2.5, self.n_samples).reshape(-1, 1, 1, 1)
        img = np.clip(img * light, 0, 1)
        return img.transpose(0, 3, 1, 2)

    def predict(self, img_1, img_2):
        imgs = np.vstack([self.preprocess_img(img_1), self.preprocess_img(img_2)])
        imgs = torch.from_numpy(imgs).float()
        pred = self.model(imgs)
        pred_1 = pred[:self.n_samples].mean(axis=0).view(3, 3, 3, 6).detach().numpy()
        pred_2 = pred[self.n_samples:].mean(axis=0).view(3, 3, 3, 6).detach().numpy()
        pred = process_preds(pred_1, pred_2)
        # print(pred)
        return pred

    

if __name__ == '__main__':
    perceiver = Perceiver()
    perceiver.get_cube_state()
    perceiver.api.send_request(move = 'reset')


