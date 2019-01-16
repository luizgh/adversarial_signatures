# Characterizing adversarial examples for Signature verification

This repository contains code to evaluate attacks against CNN-based and 
LBP-based models [1], as well as scripts to evaluate
two defense mechanisms for CNN training (Madry defense [2] and Ensemble adversarial Training [3]). 

[1] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Characterizing and evaluating adversarial examples for Offline Handwritten Signature Verification" ([preprint](https://arxiv.org/abs/1901.03398))
 
[2] Madry, A., Makelov, A., Schmidt, L., Tsipras, D. and Vladu, A., 2017. Towards deep learning models resistant to adversarial attacks. [preprint](https://arxiv.org/abs/1706.06083)

[3] Tramèr, F., Kurakin, A., Papernot, N., Goodfellow, I., Boneh, D. and McDaniel, P., 2017. Ensemble adversarial training: Attacks and defenses. [preprint](https://arxiv.org/abs/1705.07204)

Installation
============

First install the package [sigver](https://github.com/luizgh/sigver) as follows:
```bash
pip install git+https://github.com/luizgh/sigver.git --process-dependency-links
```

Download (or clone) this repository and install its requirements: 
```
pip install requirements.txt
```

For the LBP experiments, you need to have Matlab installed, and install the python/matlab integration:

https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

## Example 

An interactive example can be found in the ```example.ipynb``` file, that can also be viewed in this [link](https://nbviewer.jupyter.org/github/luizgh/adversarial_signatures/blob/master/example.ipynb). This example considers the following tasks:

1. Extracting features and training a WD classifier for a user
2. Perform a type-I attack (change a genuine signature so that it is rejected)
3. Perform a type-II attack (change a skilled forgery so that it is accepted)


## Reproducing the paper

To reproduce all steps in the paper you need access to the four datasets (GPDS, MCYT, CEDAR, Brazilian PUC-PR). 
To reproduce the result for a single dataset, you need access to this dataset, and the models 
trained on GPDS (see the [trained models section](#Trained-models)). Please note that I cannot share the datasets, so please directly 
contact the groups that proposed the datasets to have access to them (MCYT and CEDAR have public access).

To reproduce the paper, the following steps are necessary:

1. Train the CNNs on GPDS: ```scripts/train_cnns.sh``` (or use the trained models below)
2. Process the datasets - extract lbp and signet features: ```scripts/process_datasets.sh```
3. Train classifiers and select the images to be attacked (only images that are correctly classified by all models): ```scripts/choose_files.sh```
4. Run LBP attacks: ```scripts/lbp_attacks.sh```
5. Run CNN attacks: ```scripts/cnn_attacks.sh```

The scripts ```scripts/process_results.sh``` and ```scripts/process_results_[defense/otsu].py``` process the results and generate latex tables  


## Trained models

Below are the trained CNN models that were used in this paper:

* [SigNet](https://storage.googleapis.com/luizgh-datasets/pytorch_models/signet.pth) (from [4])
* [SigNet 350-615](https://storage.googleapis.com/luizgh-datasets/pytorch_models/signet_350_615.pth): trained with GPDS users [350, 615)
* [SigNet Madry 350-615](https://storage.googleapis.com/luizgh-datasets/pytorch_models/signet_350_615_madry_norm_2.pth): trained with GPDS users [350, 615) using the Madry defense [2]
* [SigNet Ens Adv 350-615](https://storage.googleapis.com/luizgh-datasets/pytorch_models/signet_350_615_ensadv_norm5.pth): trained with GPDS users [350, 615) using Ensemble Adversarial Training [3]
* [SigNet 615-881](https://storage.googleapis.com/luizgh-datasets/pytorch_models/signet_615_881.pth): trained with GPDS users [615, 881)

[4] Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks" http://dx.doi.org/10.1016/j.patcog.2017.05.012 ([preprint](https://arxiv.org/abs/1705.05787))

# Citation

If you use our code, please consider citing the following papers:

Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Learning Features for Offline Handwritten Signature Verification using Deep Convolutional Neural Networks" http://dx.doi.org/10.1016/j.patcog.2017.05.012 ([preprint](https://arxiv.org/abs/1705.05787))

Hafemann, Luiz G., Robert Sabourin, and Luiz S. Oliveira. "Characterizing and evaluating adversarial examples for Offline Handwritten Signature Verification" ([preprint](https://arxiv.org/abs/1901.03398))


# License

The source code on the project root and the "scripts" folder is released under the BSD 3-clause license.
The code under "clbp" is copyrighted by the authors Guo, Zhenhua, Lei Zhang, and David Zhang [5]. 
Note that the trained models used the GPDS dataset for training, which is restricted for non-commercial use.  


[5] Guo, Zhenhua, Lei Zhang, and David Zhang. "A completed modeling of local binary pattern operator for texture classification." 
IEEE Transactions on Image Processing 19.6 (2010): 1657-1663.