import argparse
import os
from skimage.io import imsave

from sigver.datasets.util import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', required=True)
parser.add_argument('--save-path', required=True)

args = parser.parse_args()

data = load_dataset(args.data_path)
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

x, _, _, _, filenames = data

for img, filename in tqdm(zip(x, filenames), total=len(x)):
    full_name = os.path.join(args.save_path, filename)
    imsave(full_name, 255 - img.squeeze())