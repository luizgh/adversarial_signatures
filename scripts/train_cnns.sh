#!/usr/bin/env bash

# Training the CNNs used in the work:

# Signet baselines

python -m sigver.featurelearning.train --dataset-path  ~/Datasets/gpds_170_242.npz  \
    --epochs 60 --logdir ~/runs/signet --batch-size 32 \
    --model signet

python -m sigver.wd.test -m signet --model-path ~/runs/signet/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/signet/wd_gpds300_ngen_12.pickle \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12

python -m sigver.featurelearning.train --dataset-path  ~/Datasets/gpds_170_242.npz  \
    --epochs 60 --logdir ~/runs/signet_f_lamb0.95 --batch-size 32 \
    --forg --lamb 0.95 --model signet

python -m sigver.wd.test -m signet --model-path ~/runs/signet_f_lamb0.95/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/signet_f_lamb0.95/wd_gpds300_ngen_12.pickle \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12

python -m sigver.featurelearning.train --dataset-path  ~/Datasets/gpds_170_242.npz  \
    --epochs 60 --logdir ~/runs/signet_f_lamb0.999 --batch-size 32 \
    --forg --lamb 0.999 --model signet

python -m sigver.wd.test -m signet --model-path ~/runs/signet_f_lamb0.999/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/signet_f_lamb0.999/wd_gpds300_ngen_12.pickle \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12


# Training with part of GPDS
python -m sigver.featurelearning.train --dataset-path  ~/Datasets/gpds_170_242.npz  \
    --epochs 60 --logdir ~/runs/signet_350_615 --batch-size 32 \
    --model signet --users 350 615

python -m sigver.featurelearning.train --dataset-path  ~/Datasets/gpds_170_242.npz  \
    --epochs 60 --logdir ~/runs/signet_615_881 --batch-size 32 \
    --model signet --users 615 881


python -m sigver.wd.test -m signet --model-path ~/runs/signet_350_615/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/signet_350_615/wd_gpds300_ngen_12.pickle \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12

# Other networks

python -m sigver.featurelearning.train --dataset-path  ~/Datasets/gpds_170_242.npz  \
    --epochs 60 --logdir ~/runs/signet_thin --batch-size 32 \
    --model signet_thin

python -m sigver.featurelearning.train --dataset-path  ~/Datasets/gpds_170_242.npz  \
    --epochs 60 --logdir ~/runs/signet_smaller --batch-size 32 \
    --model signet_smaller


python -m sigver.wd.test -m signet_thin --model-path ~/runs/signet_thin/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/signet_thin/wd_gpds300_ngen_12.pickle \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12

python -m sigver.wd.test -m signet_smaller --model-path ~/runs/signet_smaller/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/signet_smaller/wd_gpds300_ngen_12.pickle \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12


# Other networks on partial number of users

python -m sigver.featurelearning.train --dataset-path  ~/Datasets/gpds_170_242.npz  \
    --epochs 60 --logdir ~/runs/signet_thin_350_615 --batch-size 32 \
    --model signet_thin  --users 350 615

python -m sigver.featurelearning.train --dataset-path  ~/Datasets/gpds_170_242.npz  \
    --epochs 60 --logdir ~/runs/signet_smaller_350_615 --batch-size 32 \
    --model signet_smaller  --users 350 615


# Madry on partial # users

python -m sigver.adversarial.train_madry --dataset-path ~/Datasets/gpds_170_242.npz  \
  --epochs 60 --logdir ~/runs/signet_350_615_madry_norm_1 --batch-size 32 --model signet --attack-l2 1 \
  --users 350 615

python -m sigver.adversarial.train_madry --dataset-path ~/Datasets/gpds_170_242.npz  \
  --epochs 60 --logdir ~/runs/signet_350_615_madry_norm_2 --batch-size 32 --model signet --attack-l2 2 \
  --users 350 615

python -m sigver.adversarial.train_madry --dataset-path ~/Datasets/gpds_170_242.npz  \
  --epochs 60 --logdir ~/runs/signet_350_615_madry_norm_3 --batch-size 32 --model signet --attack-l2 3 \
  --users 350 615

python -m sigver.adversarial.train_madry --dataset-path ~/Datasets/gpds_170_242.npz  \
  --epochs 60 --logdir ~/runs/signet_350_615_madry_norm_4 --batch-size 32 --model signet --attack-l2 4 \
  --users 350 615

python -m sigver.adversarial.train_madry --dataset-path ~/Datasets/gpds_170_242.npz  \
  --epochs 60 --logdir ~/runs/signet_350_615_madry_norm_5 --batch-size 32 --model signet --attack-l2 5 \
  --users 350 615

python -m sigver.adversarial.train_madry --dataset-path ~/Datasets/gpds_170_242.npz  \
  --epochs 60 --logdir ~/runs/signet_350_615_madry_norm_10 --batch-size 32 --model signet --attack-l2 10 \
  --users 350 615

# Ensemble Adv training on partial # users

python -m sigver.adversarial.train_ensadv --dataset-path ~/Datasets/gpds_170_242.npz  \
  --epochs 60 --logdir ~/runs/signet_350_615_ensadv_norm2 --batch-size 32 --model signet --eps 2 \
  --users 350 615 --trained-models signet_thin ~/runs/signet_thin_350_615/model_last.pth signet_smaller \
  ~/runs/signet_smaller_350_615/model_last.pth


python -m sigver.adversarial.train_ensadv --dataset-path ~/Datasets/gpds_170_242.npz  \
  --epochs 60 --logdir ~/runs/signet_350_615_ensadv_norm5 --batch-size 32 --model signet --eps 5 \
  --users 350 615 --trained-models signet_thin ~/runs/signet_thin_350_615/model_last.pth signet_smaller \
  ~/runs/signet_smaller_350_615/model_last.pth


python -m sigver.adversarial.train_ensadv --dataset-path ~/Datasets/gpds_170_242.npz  \
  --epochs 60 --logdir ~/runs/signet_350_615_ensadv_norm10 --batch-size 32 --model signet --eps 10 \
  --users 350 615 --trained-models signet_thin ~/runs/signet_thin_350_615/model_last.pth signet_smaller \
  ~/runs/signet_smaller_350_615/model_last.pth


python -m sigver.adversarial.train_ensadv --dataset-path ~/Datasets/gpds_170_242.npz  \
  --epochs 60 --logdir ~/runs/signet_350_615_ensadv_norm20 --batch-size 32 --model signet --eps 20 \
  --users 350 615 --trained-models signet_thin ~/runs/signet_thin_350_615/model_last.pth signet_smaller \
  ~/runs/signet_smaller_350_615/model_last.pth


#Checking the performance for WD classification

python -m sigver.wd.test -m signet --model-path ~/runs/signet_350_615/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/signet_350_615/wd_vv_ngen_12.pickle \
    --exp-users 300 350 --dev-users 350 881 --gen-for-train 12 --folds 1

python -m sigver.wd.test -m signet --model-path ~/runs/signet_350_615_madry_norm_1/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/signet_350_615_madry_norm_1/wd_vv_ngen_12.pickle \
    --exp-users 300 350 --dev-users 350 881 --gen-for-train 12 --folds 1

python -m sigver.wd.test -m signet --model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/signet_350_615_madry_norm_2/wd_gpds300_ngen_12.pickle \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12 --folds 1

python -m sigver.wd.test -m signet --model-path ~/runs/signet_350_615_madry_norm_3/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/signet_350_615_madry_norm_3/wd_gpds300_ngen_12.pickle \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12 --folds 1

python -m sigver.wd.test -m signet --model-path ~/runs/signet_350_615_madry_norm_4/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/signet_350_615_madry_norm_4/wd_gpds300_ngen_12.pickle \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12 --folds 1

python -m sigver.wd.test -m signet --model-path ~/runs/sigver_madry_norm_5/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/sigver_madry_norm_5/wd_gpds300_ngen_12.pickle \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12 --folds 1

python -m sigver.wd.test -m signet --model-path ~/runs/sigver_madry_norm_10/model_last.pth  \
    --data-path ~/Datasets/gpds_170_242.npz --save-path ~/runs/sigver_madry_norm_10/wd_gpds300_ngen_12.pickle \
    --exp-users 0 300 --dev-users 300 881 --gen-for-train 12 --folds 1