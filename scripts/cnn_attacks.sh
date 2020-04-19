#!/usr/bin/env bash
cd ..
python cnn_attacks.py --data-path ~/dev/adversarial_signatures/data/mcyt_classifiers.pickle  \
--defense-model-path ~/dev/adversarial_signatures/models/signet.pth --save-path ~/dev/adversarial_signatures/data/mcyt_cnn_attacks_pk.pickle

python cnn_attacks.py --data-path ~/dev/adversarial_signatures/data/mcyt_classifiers.pickle  \
--defense-model-path ~/dev/adversarial_signatures/models/signet.pth --save-path ~/dev/adversarial_signatures/data/mcyt_cnn_attacks_lk.pickle --lk


python cnn_attacks.py --data-path ~/dev/adversarial_signatures/data/cedar_classifiers.pickle  \
--defense-model-path ~/dev/adversarial_signatures/models/signet.pth --save-path ~/runs/adv/cedar_cnn_attacks_pk.pickle

python cnn_attacks.py --data-path ~/dev/adversarial_signatures/data/cedar_classifiers.pickle  \
--defense-model-path ~/dev/adversarial_signatures/models/signet.pth --save-path ~/runs/adv/cedar_cnn_attacks_lk.pickle --lk

#
#
## MCYT, baseline
#
#python cnn_attacks.py --data-path ~/runs/adv/mcyt_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/mcyt_cnn_half_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/mcyt_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/mcyt_cnn_half_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/mcyt_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/mcyt_cnn_half_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
#
## MCYT madry
#
#python cnn_attacks.py --data-path ~/runs/adv/mcyt_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/mcyt_cnn_half_madry_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/mcyt_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/mcyt_cnn_half_madry_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/mcyt_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/mcyt_cnn_half_madry_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
#
## MCYT ens adv
#
#python cnn_attacks.py --data-path ~/runs/adv/mcyt_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/mcyt_cnn_half_ensadv_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/mcyt_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/mcyt_cnn_half_ensadv_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/mcyt_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/mcyt_cnn_half_ensadv_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
#
#
## cedar, baseline
#
#python cnn_attacks.py --data-path ~/runs/adv/cedar_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/cedar_cnn_half_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/cedar_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/cedar_cnn_half_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/cedar_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/cedar_cnn_half_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
#
## cedar madry
#
#python cnn_attacks.py --data-path ~/runs/adv/cedar_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/cedar_cnn_half_madry_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/cedar_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/cedar_cnn_half_madry_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/cedar_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/cedar_cnn_half_madry_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
#
## cedar ens adv
#
#python cnn_attacks.py --data-path ~/runs/adv/cedar_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/cedar_cnn_half_ensadv_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/cedar_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/cedar_cnn_half_ensadv_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/cedar_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/cedar_cnn_half_ensadv_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
#
#
## brazilian, baseline
#
#python cnn_attacks.py --data-path ~/runs/adv/brazilian_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/brazilian_cnn_half_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/brazilian_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/brazilian_cnn_half_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/brazilian_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/brazilian_cnn_half_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
#
## brazilian madry
#
#python cnn_attacks.py --data-path ~/runs/adv/brazilian_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/brazilian_cnn_half_madry_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/brazilian_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/brazilian_cnn_half_madry_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/brazilian_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/brazilian_cnn_half_madry_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
#
## brazilian ens adv
#
#python cnn_attacks.py --data-path ~/runs/adv/brazilian_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/brazilian_cnn_half_ensadv_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/brazilian_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/brazilian_cnn_half_ensadv_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/brazilian_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/brazilian_cnn_half_ensadv_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
#
#
## gpds, baseline
#
#python cnn_attacks.py --data-path ~/runs/adv/gpds_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/gpds_cnn_half_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/gpds_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/gpds_cnn_half_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/gpds_classifiers_signethalf.pickle  \
#--defense-model-path ~/runs/signet_350_615/model_last.pth --save-path ~/runs/adv/gpds_cnn_half_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
#
## gpds madry
#
#python cnn_attacks.py --data-path ~/runs/adv/gpds_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/gpds_cnn_half_madry_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/gpds_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/gpds_cnn_half_madry_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/gpds_classifiers_signethalf_madry.pickle  \
#--defense-model-path ~/runs/signet_350_615_madry_norm_2/model_last.pth --save-path ~/runs/adv/gpds_cnn_half_madry_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
#
## gpds ens adv
#
#python cnn_attacks.py --data-path ~/runs/adv/gpds_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/gpds_cnn_half_ensadv_pk.pickle
#
#python cnn_attacks.py --data-path ~/runs/adv/gpds_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/gpds_cnn_half_ensadv_lk1.pickle --lk
#
#python cnn_attacks.py --data-path ~/runs/adv/gpds_classifiers_signethalf_ensadv.pickle  \
#--defense-model-path ~/runs/signet_350_615_ensadv_norm5/model_last.pth --save-path ~/runs/adv/gpds_cnn_half_ensadv_lk2.pickle --lk \
#--attack-model-path ~/runs/signet_615_881/model_last.pth
