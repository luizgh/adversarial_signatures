#!/usr/bin/env bash

cd ../data

python process_dataset.py --dataset mcyt --path ~/Datasets/MCYT-ORIGINAL/MCYToffline75original --save-path ~/Datasets/mcyt_170_242_otsuafter.npz
python process_dataset.py --dataset cedar --path ~/Datasets/CEDAR --save-path ~/Datasets/cedar_170_242_otsuafter.npz
python process_dataset.py --dataset brazilian-nosimple --path ~/Datasets/Brazilian/signatures --save-path ~/Datasets/brazilian_170_242_otsuafter.npz
python process_dataset.py --dataset gpds --path ~/Datasets/GPDS-ORIGINAL-MOISES --save-path ~/Datasets/gpds_170_242_otsuafter.npz

cd ..

python extract_features.py --data-path ~/Datasets/mcyt_170_242_otsuafter.npz \
  --weights-path ~/runs/signet/model_last.pth --save-path-signet ~/runs/adv/features/mcyt_signet.npy \
  --save-path-lbp ~/runs/adv/features/mcyt_lbp.npy

python extract_features.py --data-path ~/Datasets/cedar_170_242_otsuafter.npz \
  --weights-path ~/runs/signet/model_last.pth --save-path-signet ~/runs/adv/features/cedar_signet.npy \
  --save-path-lbp ~/runs/adv/features/cedar_lbp.npy

python extract_features.py --data-path ~/Datasets/brazilian_170_242_otsuafter.npz \
  --weights-path ~/runs/signet/model_last.pth --save-path-signet ~/runs/adv/features/brazilian_signet.npy \
  --save-path-lbp ~/runs/adv/features/brazilian_lbp.npy

python extract_features.py --data-path ~/Datasets/gpds_170_242_otsuafter.npz \
  --weights-path ~/runs/signet/model_last.pth --save-path-signet ~/runs/adv/features/gpds_signet.npy \
  --save-path-lbp ~/runs/adv/features/gpds_lbp.npy
