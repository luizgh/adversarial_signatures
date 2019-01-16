import argparse
import numpy as np
import torch
from sigver.preprocessing.normalize import crop_center

from clbp.clbp import CLBP
from sigver.datasets.util import load_dataset
from sigver.featurelearning.data import extract_features
from sigver.featurelearning.models.signet import SigNet
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process dataset and extract SigNet and CLBP features')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--weights-path', required=True)
    parser.add_argument('--save-path-signet', required=True)
    parser.add_argument('--save-path-lbp', required=True)

    args = parser.parse_args()

    # Load dataset
    print('Loading data')
    x, y, yforg, user_mapping, filenames = load_dataset(args.data_path)
    print(x.shape)

    # Load and initialize model
    print('Extracting SigNet features')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict, class_weights, forg_weights = torch.load(args.weights_path,
                                                         map_location=lambda storage, loc: storage)

    model = SigNet()
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    def process_fn(batch):
        input = batch[0].to(device)
        return model(input)


    batch_size = 32
    input_size = (150, 220)
    cnn_features = extract_features(x, process_fn, batch_size, input_size)

    print('Extracting CLBP features')
    descriptor = CLBP()
    lbp_features = []
    for img in tqdm(x):
        img = crop_center(img.squeeze(), input_size)
        lbp_features.append(descriptor(img))
    lbp_features = np.concatenate(lbp_features)

    print(lbp_features.shape)

    np.save(args.save_path_signet, cnn_features)
    np.save(args.save_path_lbp, lbp_features)

