import sklearn
import sklearn.svm
import numpy as np
import metrics


def create_trainset_for_user(xfeatures, y, yforg, user):
    positive_samples = xfeatures[(y == user) & (yforg == 0)]
    negative_samples = xfeatures[(y != user) & (yforg == 0)]

    X = np.concatenate((positive_samples, negative_samples))
    y = np.concatenate((np.ones(len(positive_samples)), np.ones(len(negative_samples)) * -1))

    return X, y


def split_devset(y, *arr, n_users_in_first_set, rng=np.random.RandomState()):
    users = sorted(np.unique(y))

    rng.shuffle(users)

    system_users = sorted(users[0:n_users_in_first_set])

    mask = np.logical_or.reduce([y == c for c in system_users])

    arrays = [y] + list(arr)

    return (tuple(a[mask] for a in arrays),
            tuple(a[~mask] for a in arrays))


def split_train_test(y, yforg, *arr, n_train_samples, rng=np.random.RandomState()):
    users = np.unique(y)

    train_mask = np.zeros_like(y, dtype=bool)
    adv_train_mask = np.zeros_like(y, dtype=bool)

    for user in users:
        users_gen_idx = np.flatnonzero((y == user) & (yforg == 0))
        rng.shuffle(users_gen_idx)

        selected_for_train = users_gen_idx[0:n_train_samples]
        selected_for_train_adv = users_gen_idx[n_train_samples:n_train_samples * 2]

        train_mask[selected_for_train] = True
        adv_train_mask[selected_for_train_adv] = True

        assert len(set(selected_for_train).intersection(set(selected_for_train_adv))) == 0

    arrays = [y, yforg] + list(arr)
    return (tuple(a[train_mask] for a in arrays),
            tuple(a[adv_train_mask] for a in arrays),
            tuple(a[(~train_mask) & (~adv_train_mask)] for a in arrays))


def train_wdclassifier_user(svmType, C, gamma, trainingSet):
    # For the SVM training, we want a balanced dataset. One way to accomplish this is to have different
    # weights "C" for the positive and negative classes. An equivalent alternative is to duplicate the
    # genuine signatures, so they match the same number of forgeries

    assert svmType in ['linear', 'rbf']

    trainX = trainingSet[0]
    trainY = trainingSet[1]

    # compute the skew
    n_genuine = len([x for x in trainY if x == 1])
    n_forg = len([x for x in trainY if x == -1])
    skew = n_forg / float(n_genuine)

    # Normalize input (0 mean, 1 std)
    # Train the model
    if svmType == 'rbf':
        model = sklearn.svm.SVC(C=C, gamma=gamma, class_weight={1: skew})
    else:
        model = sklearn.svm.SVC(kernel='linear', C=C, class_weight={1: skew})

    model.fit(trainX, trainY)

    return model


def test_user(randomForgeries, skilledForgeries, testGenuine, model):
    testSkilledForgeries = skilledForgeries

    testGenuine = np.array(testGenuine)
    testRandomForgeries = np.array(randomForgeries)

    # Get predictions
    genuinePred = model.decision_function(testGenuine)
    randomPred = model.decision_function(testRandomForgeries)
    skilledPred = model.decision_function(testSkilledForgeries)

    return genuinePred, randomPred, skilledPred


def train_all_users(xfeatures_train, y_train, yforg_train,
                    svmType, C, gamma):
    classifier_all_user = {}

    users = np.unique(y_train)

    for user in users:
        trainingSet = create_trainset_for_user(xfeatures_train, y_train, yforg_train, user)
        classifier_all_user[user] = train_wdclassifier_user(svmType, C, gamma, trainingSet)

    return classifier_all_user


def test_all_users(classifier_all_user, xfeatures_test, y_test, yforg_test,
                   global_threshold):
    genuinePreds = []
    randomPreds = []
    skilledPreds = []

    users = np.unique(y_test)
    for user in users:
        model = classifier_all_user[user]

        # Test the performance for the user without replicates
        skilled_forgeries = xfeatures_test[(y_test == user) & (yforg_test == 1)]
        test_genuine = xfeatures_test[(y_test == user) & (yforg_test == 0)]
        random_forgeries = xfeatures_test[(y_test != user) & (yforg_test == 0)]

        genuinePredUser = model.decision_function(test_genuine)
        skilledPredUser = model.decision_function(skilled_forgeries)
        randomPredUser = model.decision_function(random_forgeries)

        genuinePreds.append(genuinePredUser)
        skilledPreds.append(skilledPredUser)
        randomPreds.append(randomPredUser)

    # Calculate al metrics (EER, FAR, FRR and AUC) decision threshold at 0 (global_threshold)
    all_metrics = metrics.calculate_metrics(global_threshold, genuinePreds, randomPreds, skilledPreds)

    results = {'all_metrics': all_metrics,
               'predictions': {'genuinePreds': genuinePreds,
                               'randomPreds': randomPreds,
                               'skilledPreds': skilledPreds}}

    print(all_metrics['EER'], all_metrics['EER_userthresholds'])
    return results


def train_test_all_users(xfeatures_train, y_train, yforg_train,
                         xfeatures_test, y_test, yforg_test,
                         svmType, C, gamma, global_threshold=0):
    classifiers_all_users = train_all_users(xfeatures_train, y_train, yforg_train,
                                            svmType, C, gamma)

    results = test_all_users(classifiers_all_users, xfeatures_test, y_test, yforg_test,
                             global_threshold)

    return results, classifiers_all_users


def train_all_users_adv(adv_xfeatures_train, adv_y_train, dev_xfeatures, dev_y, dev_yforg,
                        n_dev_for_train, svmType, C, gamma):
    dev_users = np.unique(dev_y)
    system_users = np.unique(adv_y_train)

    negative_samples = []
    for u in dev_users:
        negative_samples.append(dev_xfeatures[(dev_y == u) & (dev_yforg == 0)][0:n_dev_for_train])
    negative_samples = np.concatenate(negative_samples)

    adv_classifiers = {}

    for user in system_users:
        positive_samples = adv_xfeatures_train[adv_y_train == user]

        X = np.concatenate((positive_samples, negative_samples))
        y = np.concatenate((np.ones(len(positive_samples)), np.ones(len(negative_samples)) * -1))

        adv_classifiers[user] = train_wdclassifier_user(svmType, C, gamma, (X, y))

    return adv_classifiers

