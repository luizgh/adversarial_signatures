#!/usr/bin/env bash

cd ../data

python process_dataset.py --dataset mcyt --path ~/Datasets/MCYT-ORIGINAL/MCYToffline75original --save-path ~/Datasets/mcyt_170_242_otsuafter.npz
python process_dataset.py --dataset cedar --path ~/Datasets/CEDAR --save-path ~/Datasets/cedar_170_242_otsuafter.npz

cd ..

python extract_features.py --data-path ~/Datasets/mcyt_170_242_otsuafter.npz \
  --weights-path ~/dev/adversarial_signatures/models/signet.pth --save-path-signet ~/dev/adversarial_signatures/data/mcyt_signet.npy

python extract_features.py --data-path ~/Datasets/cedar_170_242_otsuafter.npz \
  --weights-path ~/dev/adversarial_signatures/models/signet.pth --save-path-signet ~/dev/adversarial_signatures/data//cedar_signet.npy
