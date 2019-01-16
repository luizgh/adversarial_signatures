#!/usr/bin/env bash

echo Consolidated PK
python process_results.py ~/runs/adv/mcyt_lbp_attacks_pk.pickle ~/runs/adv/cedar_lbp_attacks_pk.pickle  \
  ~/runs/adv/brazilian_lbp_attacks_pk.pickle ~/runs/adv/gpds_lbp_attacks_pk.pickle \
  ~/runs/adv/mcyt_cnn_attacks_pk.pickle ~/runs/adv/cedar_cnn_attacks_pk.pickle  \
  ~/runs/adv/brazilian_cnn_attacks_pk.pickle ~/runs/adv/gpds_cnn_attacks_pk.pickle

echo Consolidated lk
python process_results.py ~/runs/adv/mcyt_lbp_attacks_lk.pickle ~/runs/adv/cedar_lbp_attacks_lk.pickle  \
  ~/runs/adv/brazilian_lbp_attacks_lk.pickle ~/runs/adv/gpds_lbp_attacks_lk.pickle \
  ~/runs/adv/mcyt_cnn_attacks_lk.pickle ~/runs/adv/cedar_cnn_attacks_lk.pickle  \
  ~/runs/adv/brazilian_cnn_attacks_lk.pickle ~/runs/adv/gpds_cnn_attacks_lk.pickle

echo Consolidated lk2
python process_results.py ~/runs/adv/mcyt_cnn_half_lk2.pickle ~/runs/adv/cedar_cnn_half_lk2.pickle \
  ~/runs/adv/gpds_cnn_half_lk2.pickle ~/runs/adv/brazilian_cnn_half_lk2.pickle

# PK
echo MCYT PK
python process_results.py ~/runs/adv/mcyt_lbp_attacks_pk.pickle ~/runs/adv/mcyt_cnn_attacks_pk.pickle

echo CEDAR PK
python process_results.py ~/runs/adv/cedar_lbp_attacks_pk.pickle ~/runs/adv/cedar_cnn_attacks_pk.pickle

echo GPDS PK
python process_results.py ~/runs/adv/gpds_lbp_attacks_pk.pickle ~/runs/adv/gpds_cnn_attacks_pk.pickle

echo Brazilian PK
python process_results.py ~/runs/adv/brazilian_lbp_attacks_pk.pickle ~/runs/adv/brazilian_cnn_attacks_pk.pickle


# LK1
echo MCYT lk
python process_results.py ~/runs/adv/mcyt_lbp_attacks_lk.pickle ~/runs/adv/mcyt_cnn_attacks_lk.pickle

echo CEDAR lk
python process_results.py ~/runs/adv/cedar_lbp_attacks_lk.pickle ~/runs/adv/cedar_cnn_attacks_lk.pickle

echo GPDS lk
python process_results.py ~/runs/adv/gpds_lbp_attacks_lk.pickle ~/runs/adv/gpds_cnn_attacks_lk.pickle

echo Brazilian lk
python process_results.py ~/runs/adv/brazilian_lbp_attacks_lk.pickle ~/runs/adv/brazilian_cnn_attacks_lk.pickle

# LK2

echo MCYT lk2
python process_results.py  ~/runs/adv/mcyt_cnn_half_lk2.pickle

echo CEDAR lk2
python process_results.py ~/runs/adv/cedar_cnn_half_lk2.pickle

echo GPDS lk2
python process_results.py ~/runs/adv/gpds_cnn_half_lk2.pickle

echo Brazilian lk2
python process_results.py ~/runs/adv/brazilian_cnn_half_lk2.pickle
