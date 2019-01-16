#!/usr/bin/env bash

# Perfect knowledge
python lbp_attacks.py --data-path ~/runs/adv/mcyt_classifiers.pickle --save-path ~/runs/adv/mcyt_lbp_attacks_pk.pickle
python lbp_attacks.py --data-path ~/runs/adv/cedar_classifiers.pickle --save-path ~/runs/adv/cedar_lbp_attacks_pk.pickle
python lbp_attacks.py --data-path ~/runs/adv/brazilian_classifiers.pickle --save-path ~/runs/adv/brazilian_lbp_attacks_pk.pickle
python lbp_attacks.py --data-path ~/runs/adv/gpds_classifiers.pickle --save-path ~/runs/adv/gpds_lbp_attacks_pk.pickle


# Limited knowledge
python lbp_attacks.py --data-path ~/runs/adv/mcyt_classifiers.pickle --lk --save-path ~/runs/adv/mcyt_lbp_attacks_lk.pickle
python lbp_attacks.py --data-path ~/runs/adv/cedar_classifiers.pickle --lk --save-path ~/runs/adv/cedar_lbp_attacks_lk.pickle
python lbp_attacks.py --data-path ~/runs/adv/brazilian_classifiers.pickle --lk --save-path ~/runs/adv/brazilian_lbp_attacks_lk.pickle
python lbp_attacks.py --data-path ~/runs/adv/gpds_classifiers.pickle --lk --save-path ~/runs/adv/gpds_lbp_attacks_lk.pickle