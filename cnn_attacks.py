import foolbox
import torch

import numpy as np
import pickle

from sigver.featurelearning.data import extract_features
from sigver.featurelearning.models import SigNet

from attack_utils import carlini_attack, boundary_attack, anneal_attack, \
    get_score
from wd import train_all_users_adv
import argparse
from attacks.attack_utils import rmse
from attacks.fgm import fgm
from model_utils import TorchRBFSVM, TorchLinearSVM, ToTwoOutputs, \
    ModelForAnneal

# def rmse(X):
#     return np.sqrt(np.mean(np.square(X)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run adversarial attacks')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--defense-model-path', required=True)
    parser.add_argument('--attack-model-path')
    parser.add_argument('--save-path', required=True)
    parser.add_argument('--gpu-idx', default=0, type=int)
    parser.add_argument('--lk', action='store_true', dest='lk', help='Limited Knowledge scenario')

    parser.set_defaults(lk=False)
    args = parser.parse_args()
    print(args)

    if args.lk:
        if args.attack_model_path is None:
            print('No attack model informed: using the same model as the defense (LK1 scenario)')
        else:
            print('Using different CNN models for attack and defense (LK2 scenario)')

    print('Loading Data')
    rng = np.random.RandomState(1234)

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    train_set, train_set_adv = data['train_set'], data['train_set_adv']
    dev_set, test_set = data['dev_set'], data['test_set']

    classifiers, classifiers_linear = data['classifiers_cnn'], data['classifiers_cnn_linear']
    global_threshold = data['global_threshold']
    global_threshold_linear = data['global_threshold_linear']
    selected_images = data['selected_images']

    y_test, yforg_test, x_test, *_ = test_set

    print('Loading Models')
    state_dict, class_weights, forg_weights = torch.load(args.defense_model_path,
                                                         map_location=lambda
                                                         storage,
                                                         loc: storage)

    device = torch.device('cuda', args.gpu_idx) if torch.cuda.is_available() else torch.device('cpu')

    model = SigNet()
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    if args.lk and args.attack_model_path is not None:
        adv_state_dict, _, _ = torch.load(
            args.attack_model_path, map_location=lambda storage, loc: storage)

        adv_model = SigNet()
        adv_model.load_state_dict(adv_state_dict)
        adv_model = adv_model.to(device).eval()
    else:
        adv_model = model

    if args.lk:
        # Limited knowledge scenario: adversary trains its own classifiers
        print('Training classifiers for adversary')
        C = 1
        gamma = 2 ** -11

        dev_y, dev_yforg, dev_X, dev_X_features, *_ = dev_set
        adv_y_train, adv_yforg_train, adv_x_train, adv_xfeatures_train, *_ = train_set_adv

        if args.attack_model_path is not None:
            def process_fn(batch):
                # We manually divide each pixel by 255 since we are not using
                # the PIL transformation (crop was already done)
                input = batch[0].float().div(255).to(device)
                return adv_model(input)

            adv_xfeatures_train = extract_features(adv_x_train, process_fn,
                                                   batch_size=32,
                                                   input_size=None)

            dev_X_features = extract_features(dev_X, process_fn,
                                              batch_size=32,
                                              input_size=None)

        adv_classifiers = train_all_users_adv(adv_xfeatures_train,
                                              adv_y_train,
                                              dev_X_features,
                                              dev_y,
                                              dev_yforg,
                                              5, 'rbf', C, gamma)

        adv_classifiers_linear = train_all_users_adv(adv_xfeatures_train,
                                                     adv_y_train,
                                                     dev_X_features,
                                                     dev_y,
                                                     dev_yforg,
                                                     5, 'linear', C, gamma)
    else:
        # Perfect knowledge scenario: adversary has access to the actual classifiers
        adv_classifiers = classifiers
        adv_classifiers_linear = classifiers_linear

    results_genuine = []
    results_forgery = []
    rng = np.random.RandomState(1234)

    print('Starting attacks')
    for user in selected_images:
        defense_cnn_svm = torch.nn.Sequential(model,
                                      TorchRBFSVM(classifiers[user],
                                                  device)).eval()
        defense_svm_linear = torch.nn.Sequential(model, TorchLinearSVM(
            classifiers_linear[user], device)).eval()

        cnn_svm = torch.nn.Sequential(adv_model,
                                      TorchRBFSVM(adv_classifiers[user],device)).eval()
        cnn_svm_linear = torch.nn.Sequential(adv_model,
                                             TorchLinearSVM(adv_classifiers_linear[user], device)).eval()

        genuine_idx, forgery_idx, skforgery_idx = selected_images[user]

        for image_idx, image_type, result_list in [(genuine_idx, 'genuine', results_genuine),
                                      (forgery_idx, 'random', results_forgery),
                                      (skforgery_idx, 'skilled', results_forgery)]:
            if image_idx == -1:
                continue
            if image_type == 'genuine':
                target_class = 0

                def successful_attack(score, threshold):
                    return score < threshold
            else:
                target_class = 1

                def successful_attack(score, threshold):
                    return score >= threshold

            img = x_test[image_idx]

            for defense_m, m, m_name, threshold in [(defense_cnn_svm, cnn_svm, 'model_cnn_rbf', global_threshold),
                                                    (defense_svm_linear, cnn_svm_linear, 'model_cnn_linear', global_threshold_linear)]:
                print('Attacking User {}; Image type: {}, model: {}'.format(user, image_type, m_name))

                score_clean_sample = get_score(defense_m, img, device)
                # Assert that the clean sample is not adversarial
                assert not successful_attack(score_clean_sample, threshold)

                m_two_outputs = torch.nn.Sequential(m, ToTwoOutputs(threshold)).eval()

                m_foolbox = foolbox.models.PyTorchModel(m_two_outputs,
                                                        bounds=(0, 1),
                                                        device=device)

                m_anneal = ModelForAnneal(m, device)

                print('FGM')
                adv_img = fgm(m_two_outputs, img, 1000, target_class, device,
                              image_constraints=(0, 255))
                score = get_score(defense_m, adv_img, device)

                result_list.append((user, m_name, image_type, 'fgm',
                                        image_idx, adv_img,
                                        rmse(adv_img - img),
                                        score,
                                        successful_attack(score, threshold)))

                print('Carlini')
                adv_img = carlini_attack(m_two_outputs, img, target_class, device)
                score = get_score(defense_m, adv_img, device)

                result_list.append((user, m_name, image_type, 'carlini',
                                        image_idx, adv_img, rmse(adv_img - img),
                                        score, successful_attack(score, threshold)))

                print('Boundary')
                adv_img = boundary_attack(m_foolbox, img, target_class)
                if adv_img is not None:
                    score = get_score(defense_m, adv_img, device)

                    result_list.append((user, m_name, image_type,
                                            'decision', image_idx, adv_img,
                                            rmse(adv_img - img), score,
                                            successful_attack(score, threshold)))
                else:
                    result_list.append((user, m_name, image_type,
                                            'decision', image_idx, None,
                                            None, None, False))

                print('Anneal')
                adv_img = anneal_attack(m_anneal, img, threshold, target_class)
                score = get_score(defense_m, adv_img, device)

                result_list.append((user, m_name, image_type,
                                        'anneal', image_idx, adv_img,
                                        rmse(adv_img - img), score,
                                        successful_attack(score, threshold)))

        with open(args.save_path, 'wb') as f:
            pickle.dump([results_genuine, results_forgery], f)
