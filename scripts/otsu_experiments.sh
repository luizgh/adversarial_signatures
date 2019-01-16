#!/usr/bin/env bash

# CLBP
python evaluate_otsu.py --features clbp --data-path ~/runs/adv/mcyt_classifiers.pickle \
  --attacks-path  ~/runs/adv/mcyt_lbp_attacks_pk.pickle \
  --save-path ~/runs/adv/otsu/mcyt_lbp_attacks_pk_otsu.pickle
  
python evaluate_otsu.py --features clbp --data-path ~/runs/adv/cedar_classifiers.pickle \
  --attacks-path  ~/runs/adv/cedar_lbp_attacks_pk.pickle \
  --save-path ~/runs/adv/otsu/cedar_lbp_attacks_pk_otsu.pickle
  
python evaluate_otsu.py --features clbp --data-path ~/runs/adv/gpds_classifiers.pickle \
  --attacks-path  ~/runs/adv/gpds_lbp_attacks_pk.pickle \
  --save-path ~/runs/adv/otsu/gpds_lbp_attacks_pk_otsu.pickle
  
python evaluate_otsu.py --features clbp --data-path ~/runs/adv/brazilian_classifiers.pickle \
  --attacks-path  ~/runs/adv/brazilian_lbp_attacks_pk.pickle \
  --save-path ~/runs/adv/otsu/brazilian_lbp_attacks_pk_otsu.pickle


# Signet
python evaluate_otsu.py --features signet --data-path ~/runs/adv/mcyt_classifiers_signethalf.pickle \
  --attacks-path  ~/runs/adv/mcyt_cnn_half_pk.pickle \
  --weights-path ~/runs/signet_350_615/model_last.pth \
  --save-path ~/runs/adv/otsu/mcyt_cnn_half_pk_otsu.pickle
  
python evaluate_otsu.py --features signet --data-path ~/runs/adv/cedar_classifiers_signethalf.pickle \
  --attacks-path  ~/runs/adv/cedar_cnn_half_pk.pickle \
  --weights-path ~/runs/signet_350_615/model_last.pth \
  --save-path ~/runs/adv/otsu/cedar_cnn_half_pk_otsu.pickle

python evaluate_otsu.py --features signet --data-path ~/runs/adv/gpds_classifiers_signethalf.pickle \
  --attacks-path  ~/runs/adv/gpds_cnn_half_pk.pickle \
  --weights-path ~/runs/signet_350_615/model_last.pth \
  --save-path ~/runs/adv/otsu/gpds_cnn_half_pk_otsu.pickle

python evaluate_otsu.py --features signet --data-path ~/runs/adv/brazilian_classifiers_signethalf.pickle \
  --attacks-path  ~/runs/adv/brazilian_cnn_half_pk.pickle \
  --weights-path ~/runs/signet_350_615/model_last.pth \
  --save-path ~/runs/adv/otsu/brazilian_cnn_half_pk_otsu.pickle

# Signet Ens Adv
python evaluate_otsu.py --features signet --data-path ~/runs/adv/mcyt_classifiers_signethalf_ensadv.pickle \
  --attacks-path  ~/runs/adv/mcyt_cnn_half_ensadv_pk.pickle \
  --weights-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
  --save-path ~/runs/adv/otsu/mcyt_cnn_half_ensadv_pk_otsu.pickle

python evaluate_otsu.py --features signet --data-path ~/runs/adv/cedar_classifiers_signethalf_ensadv.pickle \
  --attacks-path  ~/runs/adv/cedar_cnn_half_ensadv_pk.pickle \
  --weights-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
  --save-path ~/runs/adv/otsu/cedar_cnn_half_ensadv_pk_otsu.pickle

python evaluate_otsu.py --features signet --data-path ~/runs/adv/gpds_classifiers_signethalf_ensadv.pickle \
  --attacks-path  ~/runs/adv/gpds_cnn_half_ensadv_pk.pickle \
  --weights-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
  --save-path ~/runs/adv/otsu/gpds_cnn_half_ensadv_pk_otsu.pickle
  
python evaluate_otsu.py --features signet --data-path ~/runs/adv/brazilian_classifiers_signethalf_ensadv.pickle \
  --attacks-path  ~/runs/adv/brazilian_cnn_half_ensadv_pk.pickle \
  --weights-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth \
  --save-path ~/runs/adv/otsu/brazilian_cnn_half_ensadv_pk_otsu.pickle


# Signet Madry
python evaluate_otsu.py --features signet --data-path ~/runs/adv/mcyt_classifiers_signethalf_madry.pickle \
  --attacks-path  ~/runs/adv/mcyt_cnn_half_madry_pk.pickle \
  --weights-path ~/runs/signet_350_615_madry_norm_2/model_last.pth \
  --save-path ~/runs/adv/otsu/mcyt_cnn_half_madry_pk_otsu.pickle

python evaluate_otsu.py --features signet --data-path ~/runs/adv/cedar_classifiers_signethalf_madry.pickle \
  --attacks-path  ~/runs/adv/cedar_cnn_half_madry_pk.pickle \
  --weights-path ~/runs/signet_350_615_madry_norm_2/model_last.pth \
  --save-path ~/runs/adv/otsu/cedar_cnn_half_madry_pk_otsu.pickle


python evaluate_otsu.py --features signet --data-path ~/runs/adv/gpds_classifiers_signethalf_madry.pickle \
  --attacks-path  ~/runs/adv/gpds_cnn_half_madry_pk.pickle \
  --weights-path ~/runs/signet_350_615_madry_norm_2/model_last.pth \
  --save-path ~/runs/adv/otsu/gpds_cnn_half_madry_pk_otsu.pickle


python evaluate_otsu.py --features signet --data-path ~/runs/adv/brazilian_classifiers_signethalf_madry.pickle \
  --attacks-path  ~/runs/adv/brazilian_cnn_half_madry_pk.pickle \
  --weights-path ~/runs/signet_350_615_madry_norm_2/model_last.pth \
  --save-path ~/runs/adv/otsu/brazilian_cnn_half_madry_pk_otsu.pickle

