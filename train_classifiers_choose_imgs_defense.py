import torch

import numpy as np
import pickle

from sigver.datasets.util import load_dataset, get_subset
from sigver.featurelearning.data import extract_features
from sigver.featurelearning.models import SigNet

from wd import split_train_test, split_devset, train_test_all_users
import argparse
import warnings
from sigver.preprocessing.normalize import crop_center_multiple


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run adversarial attacks')
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--models-path', required=True, nargs='*')
    parser.add_argument('--save-path', required=True, nargs='*')
    parser.add_argument('--users', default=None, nargs=2, type=int)
    parser.add_argument('--seed', default=1234)

    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # Load and split the dataset
    x, y, yforg, user_mapping, filenames = load_dataset(args.dataset_path)

    assert len(args.models_path) == len(args.save_path), 'Inform one save file for each model'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    input_size = (150, 220)
    x = crop_center_multiple(x, input_size)  # For the attacks, we will consider the inputs at size 150, 220

    all_cnn_features = []
    all_models = []
    all_classifiers = []
    all_thresholds_rbf = []
    all_classifiers_linear = []
    all_thresholds_linear = []

    print('Extracting features')
    for path in args.models_path:
        print('Using {}'.format(path))
        state_dict, class_weights, forg_weights = torch.load(path,
                                                             map_location=lambda
                                                             storage,
                                                             loc: storage)

        model = SigNet()
        model.load_state_dict(state_dict)
        model = model.to(device).eval()

        def process_fn(batch):
            # We manually divide each pixel by 255 since we are not using
            # the PIL transformation (crop was already done)
            input = batch[0].float().div(255).to(device)
            return model(input)

        cnn_features = extract_features(x, process_fn, batch_size)
        all_cnn_features.append(cnn_features)
        all_models.append(model)

    if args.users is not None:
        print('Using a subset of users from {} to {}'.format(args.users[0], args.users[1]))
        full_data = x, y, yforg, filenames, *all_cnn_features
        data = get_subset(full_data, range(args.users[0], args.users[1]), y_idx=1)
        x, y, yforg, filenames, *all_cnn_features = data

    # Split half users as "system" and half as  development (accessible
    # by the adversary)
    n_users = len(np.unique(y))
    n_exp_users = n_users // 2
    print('Using {} users for exploitation set (total: {})'.format(n_exp_users, n_users))
    exploitation_set, dev_set = split_devset(y, yforg, x,
                                             *all_cnn_features,
                                             n_users_in_first_set=n_exp_users,
                                             rng=rng)

    # Split 5 genuine signatures for training, rest for test
    exploitation_y, exploitation_yforg = exploitation_set[0:2]
    train_set, train_set_adv, test_set = split_train_test(exploitation_y,
                                                          exploitation_yforg,
                                                          *exploitation_set[2:],
                                                          n_train_samples=5,
                                                          rng=rng)

    y_train, yforg_train, x_train, *all_cnn_features_train = train_set
    y_test, yforg_test, x_test, *all_cnn_features_test = test_set

    exploitation_users = np.unique(exploitation_y)

    # Sanity check for the data
    assert len(set(exploitation_set[0]).intersection(set(dev_set[0]))) == 0
    assert len(set(exploitation_set[0]).union(set(dev_set[0]))) == n_users

    assert np.all(yforg_train == 0)
    assert len(y_train) == n_exp_users * 5

    # Train the WD classifiers
    C = 1
    gamma = 2**-11

    for model, cnn_features_train, cnn_features_test in zip(all_models,
                                                            all_cnn_features_train,
                                                            all_cnn_features_test):
        results_cnn, classifiers_cnn = train_test_all_users(cnn_features_train, y_train, yforg_train,
                                                            cnn_features_test, y_test, yforg_test,
                                                            'rbf', C, gamma)

        global_threshold = results_cnn['all_metrics']['global_threshold']

        results_cnn_linear, classifiers_cnn_linear = train_test_all_users(cnn_features_train, y_train, yforg_train,
                                                                          cnn_features_test, y_test, yforg_test,
                                                                          'linear', C, gamma)

        global_threshold_linear = results_cnn_linear['all_metrics']['global_threshold']

        all_classifiers.append(classifiers_cnn)
        all_thresholds_rbf.append(global_threshold)
        all_classifiers_linear.append(classifiers_cnn_linear)
        all_thresholds_linear.append(global_threshold_linear)

    selected_images = {}
    rng = np.random.RandomState(1234)

    for user in exploitation_users:
        rbf_models = [clf[user] for clf in all_classifiers]
        linear_models = [clf[user] for clf in all_classifiers_linear]

        # Helper functions to determine if all classifiers correctly classify a sample:
        def all_classify_as_genuine(idx):
            for m, features, t in zip(rbf_models, all_cnn_features_test,
                                      all_thresholds_rbf):
                input = np.atleast_2d(features[idx])
                if m.decision_function(input) < t:
                    return False

            for m, features, t in zip(linear_models, all_cnn_features_test,
                                      all_thresholds_linear):
                input = np.atleast_2d(features[idx])
                if m.decision_function(input) < t:
                    return False
            return True

        def all_classify_as_forgery(idx):
            for m, features, t in zip(rbf_models, all_cnn_features_test,
                                      all_thresholds_rbf):
                input = np.atleast_2d(features[idx])
                if m.decision_function(input) >= t:
                    return False

            for m, features, t in zip(linear_models, all_cnn_features_test,
                                      all_thresholds_linear):
                input = np.atleast_2d(features[idx])
                if m.decision_function(input) >= t:
                    return False
            return True

        possible_genuine_idx = np.flatnonzero((y_test == user) & (yforg_test == 0))
        possible_random_idx = np.flatnonzero((y_test != user) & (yforg_test == 0))
        possible_skforgeries_idx = np.flatnonzero((y_test == user) & (yforg_test == 1))

        # Search for a genuine signature correctly classified by all models
        genuine_idx = -1
        for idx in possible_genuine_idx:
            if all_classify_as_genuine(idx):
                genuine_idx = idx
                break
        if genuine_idx == -1:
            warnings.warn('Did not find a genuine sample for user {} correctly classified by all.'.format(user))

        # Search for a forgery correctly classified by all models
        forgery_idx = -1
        for _ in range(100):
            idx = rng.choice(possible_random_idx)
            if all_classify_as_forgery(idx):
                forgery_idx = idx
                break
        if forgery_idx == -1:
            warnings.warn('Did not find a forgery sample for user {} correctly classified by all.'.format(user))

        # Search for a genuine signature correctly classified by all models
        skforgery_idx = -1
        for idx in possible_skforgeries_idx:
            if all_classify_as_forgery(idx):
                skforgery_idx = idx
                break
        if skforgery_idx == -1:
            warnings.warn('Did not find a skilled forgery sample for user {} correctly classified by all.'.format(user))

        selected_images[user] = (genuine_idx, forgery_idx, skforgery_idx)

    # Save the results
    for i in range(len(args.save_path)):
        with open(args.save_path[i], 'wb') as f:
            # For each model, select only the CNN features extracted for that model

            y_train, yforg_train, x_train, *all_cnn_features_train = train_set
            y_test, yforg_test, x_test, *all_cnn_features_test = test_set
            dev_y, dev_yforg, dev_X, *all_dev_cnn_features = dev_set
            adv_y_train, adv_yforg_train, adv_x_train, *all_cnn_features_adv_train = train_set_adv

            user_train_set = (y_train, yforg_train, x_train, all_cnn_features_train[i])
            user_test_set = (y_test, yforg_test, x_test, all_cnn_features_test[i])
            user_dev_set = (dev_y, dev_yforg, dev_X, all_dev_cnn_features[i])
            user_train_set_adv = (adv_y_train, adv_yforg_train, adv_x_train, all_cnn_features_adv_train[i])

            to_save = {'train_set': user_train_set,
                       'train_set_adv': user_train_set_adv,
                       'test_set': user_test_set,
                       'dev_set': user_dev_set,
                       'classifiers_cnn': all_classifiers[i],
                       'classifiers_cnn_linear': all_classifiers_linear[i],
                       'global_threshold': all_thresholds_rbf[i],
                       'global_threshold_linear': all_thresholds_linear[i],
                       'selected_images': selected_images
                       }
            pickle.dump(to_save, f)
