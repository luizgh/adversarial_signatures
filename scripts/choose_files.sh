#!/usr/bin/env bash

# Choose images for attacks with LBP and CNN

python train_classifiers_choose_imgs.py --dataset-path ~/Datasets/mcyt_170_242_otsuafter.npz \
  --lbp-features-path ~/runs/adv/features/mcyt_lbp.npy \
  --signet-features-path ~/runs/adv/features/mcyt_signet.npy --save-path ~/runs/adv/mcyt_classifiers.pickle

python train_classifiers_choose_imgs.py --dataset-path ~/Datasets/cedar_170_242_otsuafter.npz \
  --lbp-features-path ~/runs/adv/features/cedar_lbp.npy \
  --signet-features-path ~/runs/adv/features/cedar_signet.npy --save-path ~/runs/adv/cedar_classifiers.pickle

python train_classifiers_choose_imgs.py --dataset-path ~/Datasets/brazilian_170_242_otsuafter.npz \
  --lbp-features-path ~/runs/adv/features/brazilian_lbp.npy \
  --signet-features-path ~/runs/adv/features/brazilian_signet.npy --save-path ~/runs/adv/brazilian_classifiers.pickle

python train_classifiers_choose_imgs.py --dataset-path ~/Datasets/gpds_170_242_otsuafter.npz \
  --lbp-features-path ~/runs/adv/features/gpds_lbp.npy \
  --signet-features-path ~/runs/adv/features/gpds_signet.npy  --users 0 160 \
  --save-path ~/runs/adv/gpds_classifiers.pickle

# Choose images for tests of different defenses (CNN trained on half; madry; ens adv)

python train_classifiers_choose_imgs_defense.py --dataset-path ~/Datasets/mcyt_170_242_otsuafter.npz \
  --models-path ~/runs/signet_350_615/model_last.pth  \
                ~/runs/signet_350_615_madry_norm_2/model_last.pth \
                ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
  --save-path ~/runs/adv/mcyt_classifiers_signethalf.pickle \
              ~/runs/adv/mcyt_classifiers_signethalf_madry.pickle \
              ~/runs/adv/mcyt_classifiers_signethalf_ensadv.pickle

python train_classifiers_choose_imgs_defense.py --dataset-path ~/Datasets/cedar_170_242_otsuafter.npz \
  --models-path ~/runs/signet_350_615/model_last.pth  \
                ~/runs/signet_350_615_madry_norm_2/model_last.pth \
                ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
  --save-path ~/runs/adv/cedar_classifiers_signethalf.pickle \
              ~/runs/adv/cedar_classifiers_signethalf_madry.pickle \
              ~/runs/adv/cedar_classifiers_signethalf_ensadv.pickle

python train_classifiers_choose_imgs_defense.py --dataset-path ~/Datasets/brazilian_170_242_otsuafter.npz \
  --models-path ~/runs/signet_350_615/model_last.pth  \
                ~/runs/signet_350_615_madry_norm_2/model_last.pth \
                ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
  --save-path ~/runs/adv/brazilian_classifiers_signethalf.pickle \
              ~/runs/adv/brazilian_classifiers_signethalf_madry.pickle \
              ~/runs/adv/brazilian_classifiers_signethalf_ensadv.pickle

python train_classifiers_choose_imgs_defense.py --dataset-path ~/Datasets/gpds_170_242_otsuafter.npz \
  --models-path ~/runs/signet_350_615/model_last.pth  \
                ~/runs/signet_350_615_madry_norm_2/model_last.pth \
                ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
  --save-path ~/runs/adv/gpds_classifiers_signethalf.pickle \
              ~/runs/adv/gpds_classifiers_signethalf_madry.pickle \
              ~/runs/adv/gpds_classifiers_signethalf_ensadv.pickle \
  --users 0 160
