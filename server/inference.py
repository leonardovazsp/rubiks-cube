from model import RubiksCubeModel
from data_pipeline import SampleGenerator
import json

config = json.load(open('config.json'))
model = RubiksCubeModel.load_from_checkpoint(config['model_path'])

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
    image = Image.fromarray(image)
    image = image.resize(self.resolution)
    image = np.array(image)/255
    light = np.linspace(0.5, 2.5, 10).reshape(-1, 1, 1, 1)
    image = image * light
    return image.transpose(0, 3, 1, 2)

def predict(img_1, img_2):
    pred_1 = model.predict(preprocess_img(img_1))
    pred_2 = model.predict(preprocess_img(img_2))
    pred = process_preds(pred_1, pred_2)
    return pred

def evaluate(img_1, img_2, label):
    pred = predict(img_1, img_2)
    diff = (pred >= 0.5) != label
    total_difference = diff.max()
    total_error = diff.sum()
    wrong_types = diff.sum((1, 2))
    return {
        'total_difference': total_difference,
        'total_error': total_error,
        'wrong_types': wrong_types
    }

img_1, img_2, labels = SampleGenerator().get_imgs()

