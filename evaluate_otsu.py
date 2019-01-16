import argparse
import torch
import numpy as np
import pickle
from sigver.featurelearning.models import SigNet
from functools import partial

from model_utils import TorchRBFSVM, TorchLinearSVM
from noise import run_otsu


def get_score_signet(model, img, device):
    input = torch.tensor(img[np.newaxis]).float().div(255).to(device)
    with torch.no_grad():
        score = model(input)
    return score.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate attack performance after OTSU')

    parser.add_argument('--features', choices=['clbp', 'signet'])

    parser.add_argument('--data-path', required=True)
    parser.add_argument('--attacks-path', required=True)
    parser.add_argument('--weights-path')
    parser.add_argument('--gpu-idx', default=0, type=int)
    parser.add_argument('--save-path', required=True)

    args = parser.parse_args()

    rng = np.random.RandomState(1234)

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    if args.features == 'clbp':
        from clbp.clbp import CLBP
        from clbp.lbp_model_utils import lbp_model

        descriptor = CLBP()
        classifiers = data['classifiers_lbp']
        classifiers_linear = data['classifiers_lbp_linear']
        global_threshold = data['global_threshold_lbp']
        global_threshold_linear = data['global_threshold_lbp_linear']

    else:
        assert args.weights_path is not None, \
            'Must inform weights-path when using signet features'

        state_dict, class_weights, forg_weights = torch.load(
            args.weights_path, map_location=lambda storage, loc: storage)

        if torch.cuda.is_available():
            device = torch.device('cuda', args.gpu_idx)
        else:
            device = torch.device('cpu')

        signetmodel = SigNet()
        signetmodel.load_state_dict(state_dict)
        signetmodel = signetmodel.to(device).eval()

        classifiers = data['classifiers_cnn']
        classifiers_linear = data['classifiers_cnn_linear']
        global_threshold = data['global_threshold']
        global_threshold_linear = data['global_threshold_linear']

    with open(args.attacks_path, 'rb') as f:
        attack_results = pickle.load(f)

    results_genuine, results_forgery = attack_results

    new_results_genuine = []
    new_results_forgery = []

    for old_results, new_results, isforgery in [[results_genuine, new_results_genuine, False],
                                                [results_forgery, new_results_forgery, True]]:
        def successful_attack(score, threshold):
            if isforgery:
                return score >= threshold
            else:
                return score < threshold

        for r in old_results:
            user = r[0]

            model_name = r[1]
            adv_img = r[5]
            score = r[7]
            success_without_otsu = r[8]

            if args.features == 'clbp':
                if 'rbf' in model_name:
                    model = lbp_model(descriptor, classifiers[user],
                                      global_threshold)
                    threshold = global_threshold
                else:
                    model = lbp_model(descriptor, classifiers_linear[user],
                                          global_threshold_linear)
                    threshold = global_threshold_linear
                get_score = model.predict_score
            else:
                if 'rbf' in model_name:
                    model = torch.nn.Sequential(signetmodel, TorchRBFSVM(
                        classifiers[user], device)).eval()
                    threshold = global_threshold
                else:
                    model = torch.nn.Sequential(signetmodel,
                                                TorchLinearSVM(classifiers_linear[user],
                                                    device)).eval()
                    threshold = global_threshold_linear
                get_score = partial(get_score_signet, model, device=device)

            if success_without_otsu:
                assert np.allclose(score, get_score(adv_img))

                # Corner case: otsu fails if image is all 0s:
                if np.all(adv_img == 0):
                    print('Warning: adversarial is all zeros (may happen for carlini).'
                          'User: {}, model_name: {}, score: {}'.format(user, model_name, score))
                    after_otsu = None
                    score_otsu = score
                    result_otsu = success_without_otsu
                else:
                    after_otsu = run_otsu(adv_img)
                    score_otsu = get_score(after_otsu)
                    result_otsu = successful_attack(score_otsu, threshold)
                new_results.append(list(r) + [after_otsu, score_otsu, result_otsu])
            else:
                new_results.append(list(r) + [None, None, False])

    with open(args.save_path, 'wb') as f:
        pickle.dump([new_results_genuine, new_results_forgery], f)
