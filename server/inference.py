from model import RubiksCubeModel
from data_pipeline import SampleGenerator
import json
import torch
import numpy as np
from PIL import Image
import copy
import random

config = json.load(open('config.json'))
resolution = config['train']['input_shape'][1:]
model = RubiksCubeModel()
model.load_state_dict(torch.load(config['model_path']))


def process_preds(pred_1, pred_2):
    # Join two labels together (inverse of pre_process_label function)
    pred_2_ = copy.deepcopy(pred_2)
    pred_2[0] = np.rot90(pred_2_[1], 2)
    pred_2[1] = np.rot90(pred_2_[0], 2)
    pred_2[2] = np.rot90(pred_2_[2], -1)
    pred = np.zeros(shape=(6, 3, 3))
    pred[[0, 1, 4]] = pred_1
    pred[[2, 3, 5]] = pred_2
    return pred

def preprocess_img(img):
    img = Image.fromarray(img)
    img = img.resize(resolution)
    img = np.array(img)/255
    light = np.linspace(1, 2.5, 10).reshape(-1, 1, 1, 1)
    img = np.clip(img * light, 0, 1)
    return img.transpose(0, 3, 1, 2)

def predict(img_1, img_2):
    imgs = np.vstack([preprocess_img(img_1), preprocess_img(img_2)])
    imgs = torch.from_numpy(imgs).float()
    pred = model(imgs)
    pred_1 = pred[:10].mean(axis=0).argmax(-1, keepdims=False).view(3, 3, 3).numpy()
    pred_2 = pred[10:].mean(axis=0).argmax(-1, keepdims=False).view(3, 3, 3).numpy()
    pred = process_preds(pred_1, pred_2)
    print(pred)
    return pred

def evaluate(img_1, img_2, label):
    pred = predict(img_1, img_2)
    acc = (pred == label).sum() / 54
    return acc

if __name__ == '__main__':
    img_1, img_2, labels = SampleGenerator().get_imgs()
    acc = evaluate(img_1, img_2, labels)
    print(acc)

