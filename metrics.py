import sklearn.metrics
import sklearn
from sklearn.metrics import roc_auc_score
import numpy as np


def calculate_metrics(global_threshold, genuinePreds, randomPreds, skilledPreds):

    allRealPreds = np.concatenate(genuinePreds)
    allRandomPreds = np.concatenate(randomPreds)
    allSkilledPreds = np.concatenate(skilledPreds)

    print('Error rates:')

    FRR = 1 - np.mean(allRealPreds >= global_threshold)
    FAR_random = 1 - np.mean(allRandomPreds < global_threshold)
    FAR_skilled = 1 - np.mean(allSkilledPreds < global_threshold)

    print('FRR: ', FRR * 100)
    print('FAR_random', FAR_random * 100)
    print('FAR_skilled', FAR_skilled * 100)

    aucs, meanAUC = calculate_AUCs(genuinePreds, skilledPreds)
    print('Mean AUC: ', meanAUC)

    EER, global_threshold = calculate_EER(allRealPreds, allSkilledPreds)
    EER_userthresholds = calculate_EER_user_thresholds(genuinePreds, skilledPreds)
    print('Equal error rate: ', EER * 100)
    print('Global threshold: ', global_threshold)
    print('EER user-based: ', EER_userthresholds * 100)

    all_metrics = {'FRR': FRR,
            'FAR_random': FAR_random,
            'FAR_skilled': FAR_skilled,
            'mean_AUC': meanAUC,
            'EER': EER,
            'EER_userthresholds': EER_userthresholds,
            'auc_list': aucs,
            'global_threshold': global_threshold}

    return all_metrics


def calculate_AUCs(genuinePreds, skilledPreds):
    aucs = []
    for thisRealPreds, thisSkilledPreds in zip(genuinePreds, skilledPreds):
        y_true = np.ones(len(thisRealPreds) + len(thisSkilledPreds))
        y_true[len(thisRealPreds):] = -1
        y_scores = np.concatenate([thisRealPreds, thisSkilledPreds])
        aucs.append(roc_auc_score(y_true, y_scores))
    meanAUC = np.mean(aucs)
    return aucs, meanAUC


def calculate_EER(allRealPreds, allSkilledPreds):
    # Calculate Equal Error Rate with a global decision threshold.

    allPreds = np.concatenate([allRealPreds, allSkilledPreds])
    allYs = np.concatenate([np.ones_like(allRealPreds), np.ones_like(allSkilledPreds) * -1])
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(allYs, allPreds)

    # Select the threshold closest to (FPR = 1 - TPR).
    t = thresholds[sorted(enumerate(abs(fpr - (1 - tpr))), key=lambda x: x[1])[0][0]]
    genuineErrors = 1 - np.mean(allRealPreds >= t)
    skilledErrors = 1 - np.mean(allSkilledPreds < t)
    EER = (genuineErrors + skilledErrors) / 2.0
    return EER, t


def calculate_EER_user_thresholds(genuinePreds, skilledPreds):

    # Calculate Equal Error Rate with a decision threshold specific for each user
    allgenuineErrors = []
    allskilledErrors = []

    nRealPreds = 0
    nSkilledPreds = 0

    for thisRealPreds, thisSkilledPreds in zip(genuinePreds, skilledPreds):
        # Calculate user AUC
        y_true = np.ones(len(thisRealPreds) + len(thisSkilledPreds))
        y_true[len(thisRealPreds):] = -1
        y_scores = np.concatenate([thisRealPreds, thisSkilledPreds])

        # Calculate user threshold
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_scores)
        # Select the threshold closest to (FPR = 1 - TPR).
        t = thresholds[sorted(enumerate(abs(fpr - (1 - tpr))), key=lambda x: x[1])[0][0]]

        genuineErrors = np.sum(thisRealPreds < t)
        skilledErrors = np.sum(thisSkilledPreds >= t)

        allgenuineErrors.append(genuineErrors)
        allskilledErrors.append(skilledErrors)

        nRealPreds += len(thisRealPreds)
        nSkilledPreds += len(thisSkilledPreds)

    genuineErrors = float(np.sum(allgenuineErrors)) / nRealPreds
    skilledErrors = float(np.sum(allskilledErrors)) / nSkilledPreds

    # Errors should be nearly equal, up to a small rounding error since we have few examples per user.
    EER = (genuineErrors + skilledErrors) / 2.0
    return EER
