import numpy as np
import pickle

from sigver.datasets.util import load_dataset, get_subset

from wd import split_train_test, split_devset, train_test_all_users
import argparse
import warnings
from sigver.preprocessing.normalize import crop_center_multiple


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run adversarial attacks')
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--signet-features-path', required=True)
    parser.add_argument('--save-path', required=True)
    parser.add_argument('--users', default=None, nargs=2, type=int)
    parser.add_argument('--seed', default=1234)

    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # Load and split the dataset
    x, y, yforg, user_mapping, filenames = load_dataset(args.dataset_path)
    cnn_features = np.load(args.signet_features_path)

    if args.users is not None:
        print('Using a subset of users from {} to {}'.format(args.users[0], args.users[1]))
        full_data = x, y, yforg, filenames, cnn_features
        data = get_subset(full_data, range(args.users[0], args.users[1]), y_idx=1)
        x, y, yforg, filenames, cnn_features = data

    x = crop_center_multiple(x, (150, 220))  # For the attacks, we will consider the inputs at size 150, 220

    # Split half users as "system" and half as  development (accessible by the adversary)
    n_users = len(np.unique(y))
    n_exp_users = n_users // 2
    print('Using {} users for exploitation set (total: {})'.format(n_exp_users, n_users))
    exploitation_set, dev_set = split_devset(y, yforg, x,
                                             cnn_features,
                                             n_users_in_first_set=n_exp_users,
                                             rng=rng)

    # Split 5 genuine signatures for training, rest for test
    exploitation_y, exploitation_yforg = exploitation_set[0:2]
    train_set, train_set_adv, test_set = split_train_test(exploitation_y,
                                                          exploitation_yforg,
                                                          *exploitation_set[2:],
                                                          n_train_samples=5,
                                                          rng=rng)

    y_train, yforg_train, x_train, cnn_features_train = train_set
    y_test, yforg_test, x_test, cnn_features_test = test_set

    exploitation_users = np.unique(exploitation_y)

    # Sanity check for the data
    assert len(set(exploitation_set[0]).intersection(set(dev_set[0]))) == 0
    assert len(set(exploitation_set[0]).union(set(dev_set[0]))) == n_users

    assert np.all(yforg_train == 0)
    assert len(y_train) == n_exp_users * 5

    # Train the WD classifiers
    C = 1
    gamma = 2**-11

    results_cnn, classifiers_cnn = train_test_all_users(cnn_features_train, y_train, yforg_train,
                                                        cnn_features_test, y_test, yforg_test,
                                                        'rbf', C, gamma)

    global_threshold = results_cnn['all_metrics']['global_threshold']

    results_cnn_linear, classifiers_cnn_linear = train_test_all_users(cnn_features_train, y_train, yforg_train,
                                                                      cnn_features_test, y_test, yforg_test,
                                                                      'linear', C, gamma)

    global_threshold_linear = results_cnn_linear['all_metrics']['global_threshold']

    gamma_lbp = 2 ** -15

    selected_images = {}
    rng = np.random.RandomState(1234)

    for user in exploitation_users:
        cnn_models = [classifiers_cnn[user], classifiers_cnn_linear[user]]
        cnn_thresholds = [global_threshold, global_threshold_linear]

        # Helper functions to determine if all classifiers correctly classify a sample:

        def all_classify_as_genuine(cnn_feature):
            cnn_feature = np.atleast_2d(cnn_feature)
            return np.all([m.decision_function(cnn_feature) >= t for m, t in zip(cnn_models, cnn_thresholds)])

        def all_classify_as_forgery(cnn_feature):
            cnn_feature = np.atleast_2d(cnn_feature)
            return np.all([m.decision_function(cnn_feature) < t for m, t in zip(cnn_models, cnn_thresholds)])

        possible_genuine_idx = np.flatnonzero((y_test == user) & (yforg_test == 0))
        possible_random_idx = np.flatnonzero((y_test != user) & (yforg_test == 0))
        possible_skforgeries_idx = np.flatnonzero((y_test == user) & (yforg_test == 1))

        # Search for a genuine signature correctly classified by all models
        genuine_idx = -1
        for idx in possible_genuine_idx:
            if all_classify_as_genuine(cnn_features_test[idx]):
                genuine_idx = idx
                break
        if genuine_idx == -1:
            warnings.warn('Did not find a genuine sample for user {} correctly classified by all.'.format(user))

        # Search for a forgery correctly classified by all models
        forgery_idx = -1
        for _ in range(100):
            idx = rng.choice(possible_random_idx)
            if all_classify_as_forgery(cnn_features_test[idx]):
                forgery_idx = idx
                break
        if forgery_idx == -1:
            warnings.warn('Did not find a forgery sample for user {} correctly classified by all.'.format(user))

        # Search for a genuine signature correctly classified by all models
        skforgery_idx = -1
        for idx in possible_skforgeries_idx:
            if all_classify_as_forgery(cnn_features_test[idx]):
                skforgery_idx = idx
                break
        if skforgery_idx == -1:
            warnings.warn('Did not find a skilled forgery sample for user {} correctly classified by all.'.format(user))

        selected_images[user] = (genuine_idx, forgery_idx, skforgery_idx)

    # Save the results
    with open(args.save_path, 'wb') as f:
        to_save = {'train_set': train_set,
                   'train_set_adv': train_set_adv,
                   'test_set': test_set,
                   'dev_set': dev_set,
                   'classifiers_cnn': classifiers_cnn,
                   'classifiers_cnn_linear': classifiers_cnn_linear,
                   'global_threshold': global_threshold,
                   'global_threshold_linear': global_threshold_linear,
                   'selected_images': selected_images
                   }
        pickle.dump(to_save, f)
