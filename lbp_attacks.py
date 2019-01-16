import numpy as np
import pickle
from clbp.clbp import CLBP
from wd import train_all_users_adv
from foolbox.attacks.boundary_attack import BoundaryAttack
from foolbox.criteria import TargetClass
from clbp.lbp_model_utils import lbp_model
from attacks.anneal import AdversaryAttackProblem
import argparse
from attacks.attack_utils import rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run adversarial attacks')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--save-path', required=True)
    parser.add_argument('--lk', action='store_true', dest='lk', help='Limited Knowledge scenario')

    parser.set_defaults(lk=False)
    args = parser.parse_args()
    print(args)

    rng = np.random.RandomState(1234)

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    train_set, train_set_adv = data['train_set'], data['train_set_adv']
    dev_set, test_set = data['dev_set'], data['test_set']

    classifiers_lbp, classifiers_lbp_linear = data['classifiers_lbp'], data['classifiers_lbp_linear']
    global_threshold_lbp = data['global_threshold_lbp']
    global_threshold_lbp_linear = data['global_threshold_lbp_linear']
    selected_images = data['selected_images']

    y_test, yforg_test, x_test, xfeatures_test, xfeature_lbp_test = test_set

    descriptor = CLBP()

    if args.lk:
        # Limited knowledge scenario: adversary trains its own classifiers
        C = 1
        gamma = 2 ** -11
        gamma_lbp = 2 ** -15

        dev_y, dev_yforg, dev_X, dev_X_features, dev_X_features_lbp = dev_set
        adv_y_train, adv_yforg_train, adv_x_train, adv_xfeatures_train, adv_xfeatures_lbp_train = train_set_adv

        adv_classifiers_lbp = train_all_users_adv(adv_xfeatures_lbp_train,
                                                  adv_y_train, dev_X_features_lbp,
                                                  dev_y, dev_yforg,
                                                  5, 'rbf', C, gamma_lbp)

        adv_classifiers_lbp_linear = train_all_users_adv(adv_xfeatures_lbp_train, adv_y_train, dev_X_features_lbp, dev_y,
                                                         dev_yforg,
                                                         5, 'linear', C, gamma_lbp)
    else:
        # Perfect knowledge scenario: adversary has access to the actual classifiers
        adv_classifiers_lbp = classifiers_lbp
        adv_classifiers_lbp_linear = classifiers_lbp_linear

    results_genuine = []
    results_forgery = []
    rng = np.random.RandomState(1234)

    for user in selected_images:
        print('Attacking user: %d' % user)
        model_lbp_rbf = lbp_model(descriptor, classifiers_lbp[user], global_threshold_lbp)
        model_lbp_linear = lbp_model(descriptor, classifiers_lbp_linear[user], global_threshold_lbp_linear)

        adv_model_lbp_rbf = lbp_model(descriptor, adv_classifiers_lbp[user], global_threshold_lbp)
        adv_model_lbp_linear = lbp_model(descriptor, adv_classifiers_lbp_linear[user], global_threshold_lbp_linear)

        models = [model_lbp_rbf, model_lbp_linear]
        adv_models = [adv_model_lbp_rbf, adv_model_lbp_linear]
        modelnames = ['model_lbp_rbf', 'model_lbp_linear']
        thresholds = [global_threshold_lbp, global_threshold_lbp_linear]

        genuine_idx, forgery_idx, skforgery_idx = selected_images[user]

        # Attack genuine images
        if genuine_idx != -1:
            selected_genuine = x_test[genuine_idx].squeeze()
            for original_m, adv_m, mname, t in zip(models, adv_models, modelnames, thresholds):
                assert original_m.predictions(selected_genuine)[1] == 1
                atk = BoundaryAttack(adv_m, TargetClass(0))

                print('Running Boundary attack on {}'.format(mname))
                boundary_result = atk(selected_genuine.astype(np.float32), 1, iterations=1000, verbose=False)

                if boundary_result is not None:
                    results_genuine.append((user, mname, 'genuine', 'decision', genuine_idx, boundary_result,
                                            rmse(boundary_result - selected_genuine),
                                            original_m.predict_score(boundary_result),
                                            original_m.predictions(boundary_result)[0]))
                else:
                    results_genuine.append((user, mname, 'genuine', 'decision', genuine_idx, boundary_result,
                                            None,
                                            None,
                                            0))

                optim = AdversaryAttackProblem(selected_genuine, adv_m,
                                               multiplier=1, norm_weight=1. / 100,
                                               threshold=t,
                                               early_stop=True,
                                               std=0.5)

                optim.steps = 1000
                optim.copy_strategy = 'slice'
                optim.Tmax = 1
                optim.Tmin = 0.001
                optim.updates = 100

                print('Running Anneal %s' % mname)
                anneal_result, e = optim.anneal()

                results_genuine.append((user, mname, 'genuine', 'anneal', genuine_idx, anneal_result,
                                        rmse(anneal_result - selected_genuine),
                                        original_m.predict_score(anneal_result),
                                        original_m.predictions(anneal_result)[0]))

        if skforgery_idx != -1:
            selected_skilled_forgery = x_test[skforgery_idx].squeeze()
            # Create gradient-based attacks for cnn models

            for original_m, adv_m, mname, t in zip(models, adv_models, modelnames, thresholds):
                assert original_m.predictions(selected_skilled_forgery)[0] == 1
                atk = BoundaryAttack(adv_m, TargetClass(1))

                boundary_result = atk(selected_skilled_forgery.astype(np.float32), 0, iterations=1000, verbose=False)

                if boundary_result is not None:
                    results_forgery.append((user, mname, 'skilled', 'decision', skforgery_idx, boundary_result,
                                            rmse(boundary_result - selected_skilled_forgery),
                                            original_m.predict_score(boundary_result),
                                            original_m.predictions(boundary_result)[1]))
                else:
                    results_forgery.append((user, mname, 'skilled', 'decision', skforgery_idx, boundary_result,
                                            None,
                                            None,
                                            0))

                optim = AdversaryAttackProblem(selected_skilled_forgery, adv_m,
                                               multiplier=-1, norm_weight=1. / 100,
                                               threshold=t,
                                               early_stop=True,
                                               std=0.5)

                optim.steps = 1000
                optim.copy_strategy = 'slice'
                optim.Tmax = 1
                optim.Tmin = 0.001
                optim.updates = 100

                anneal_result, e = optim.anneal()

                results_forgery.append((user, mname, 'skilled', 'anneal', skforgery_idx, anneal_result,
                                        rmse(anneal_result - selected_skilled_forgery),
                                        original_m.predict_score(anneal_result),
                                        original_m.predictions(anneal_result)[1]))

        if forgery_idx != -1:
            selected_random_forgery = x_test[forgery_idx].squeeze()
            # Create gradient-based attacks for cnn models

            for original_m, adv_m, mname, t in zip(models, adv_models, modelnames, thresholds):
                assert original_m.predictions(selected_random_forgery)[0] == 1
                print('Running Boundary %s; forgery' % mname)
                atk = BoundaryAttack(adv_m, TargetClass(1))

                boundary_result = atk(selected_random_forgery.astype(np.float32), 0, iterations=1000, verbose=False)

                if boundary_result is not None:
                    results_forgery.append((user, mname, 'random', 'decision', forgery_idx, boundary_result,
                                            rmse(boundary_result - selected_random_forgery),
                                            original_m.predict_score(boundary_result),
                                            original_m.predictions(boundary_result)[1]))
                else:
                    results_forgery.append((user, mname, 'random', 'decision', forgery_idx, boundary_result,
                                            None,
                                            None,
                                            0))

                optim = AdversaryAttackProblem(selected_random_forgery, adv_m,
                                               multiplier=-1, norm_weight=1. / 100,
                                               threshold=t,
                                               early_stop=True,
                                               std=0.5)

                optim.steps = 1000
                optim.copy_strategy = 'slice'
                optim.Tmax = 1
                optim.Tmin = 0.001
                optim.updates = 100

                print('Running Anneal %s; forgery' % mname)
                anneal_result, e = optim.anneal()

                results_forgery.append((user, mname, 'random', 'anneal', forgery_idx, anneal_result,
                                        rmse(anneal_result - selected_random_forgery),
                                        original_m.predict_score(anneal_result),
                                        original_m.predictions(anneal_result)[1]))

        with open(args.save_path, 'wb') as f:
            pickle.dump([results_genuine, results_forgery], f)

    descriptor.close()
