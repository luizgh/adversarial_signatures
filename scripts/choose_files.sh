#!/usr/bin/env bash

# Choose images for attacks with LBP and CNN
cd ..

python train_classifiers_choose_imgs.py --dataset-path ~/Datasets/mcyt_170_242_otsuafter.npz \
  --signet-features-path ~/dev/adversarial_signatures/data/mcyt_signet.npy --save-path ~/dev/adversarial_signatures/data/mcyt_classifiers.pickle

python train_classifiers_choose_imgs.py --dataset-path ~/Datasets/cedar_170_242_otsuafter.npz \
  --signet-features-path ~/dev/adversarial_signatures/data//cedar_signet.npy --save-path ~/dev/adversarial_signatures/data/cedar_classifiers.pickle


# Choose images for tests of different defenses (CNN trained on half; madry; ens adv)

#python train_classifiers_choose_imgs_defense.py --dataset-path ~/Datasets/mcyt_170_242_otsuafter.npz \
#  --models-path ~/runs/signet_350_615/model_last.pth  \
#                ~/runs/signet_350_615_madry_norm_2/model_last.pth \
#                ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
#  --save-path ~/dev/adversarial_signatures/data//mcyt_classifiers_signethalf.pickle \
#              ~/dev/adversarial_signatures/data//mcyt_classifiers_signethalf_madry.pickle \
#              ~/dev/adversarial_signatures/data//mcyt_classifiers_signethalf_ensadv.pickle
#
#python train_classifiers_choose_imgs_defense.py --dataset-path ~/Datasets/cedar_170_242_otsuafter.npz \
#  --models-path ~/runs/signet_350_615/model_last.pth  \
#                ~/runs/signet_350_615_madry_norm_2/model_last.pth \
#                ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
#  --save-path ~/dev/adversarial_signatures/data//cedar_classifiers_signethalf.pickle \
#              ~/dev/adversarial_signatures/data//cedar_classifiers_signethalf_madry.pickle \
#              ~/dev/adversarial_signatures/data//cedar_classifiers_signethalf_ensadv.pickle
#
#python train_classifiers_choose_imgs_defense.py --dataset-path ~/Datasets/brazilian_170_242_otsuafter.npz \
#  --models-path ~/runs/signet_350_615/model_last.pth  \
#                ~/runs/signet_350_615_madry_norm_2/model_last.pth \
#                ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
#  --save-path ~/dev/adversarial_signatures/data//brazilian_classifiers_signethalf.pickle \
#              ~/dev/adversarial_signatures/data//brazilian_classifiers_signethalf_madry.pickle \
#              ~/dev/adversarial_signatures/data//brazilian_classifiers_signethalf_ensadv.pickle
#
#python train_classifiers_choose_imgs_defense.py --dataset-path ~/Datasets/gpds_170_242_otsuafter.npz \
#  --models-path ~/runs/signet_350_615/model_last.pth  \
#                ~/runs/signet_350_615_madry_norm_2/model_last.pth \
#                ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
#  --save-path ~/dev/adversarial_signatures/data//gpds_classifiers_signethalf.pickle \
#              ~/dev/adversarial_signatures/data//gpds_classifiers_signethalf_madry.pickle \
#              ~/dev/adversarial_signatures/data//gpds_classifiers_signethalf_ensadv.pickle \
#  --users 0 160
