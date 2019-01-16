import tensorflow as tf
from cleverhans.attacks_tf import fgm
import numpy as np

import sys
sys.path.append('nn_robust_attacks')
from l2_attack import CarliniL2
from li_attack import CarliniLi


def rmse(X):
    return np.sqrt(np.mean(np.square(X)))


class FGMAttack:
    def __init__(self, sess, model_input, model_result, global_threshold, ord):
        self.sess = sess
        self.model_result_2classes = tf.concat((global_threshold - model_result,
                                                model_result - global_threshold), axis=1)
        self.model_input = model_input
        self.eps = tf.placeholder(dtype=tf.float32)
        self.labels = tf.placeholder(dtype=tf.float32)
        self.adv = fgm(self.model_input, self.model_result_2classes, self.labels,
                       eps=self.eps, clip_min=0, clip_max=255, ord=ord, targeted=True)

    def attack(self, X, target_labels, eps):
        labels_onehot = one_hot(target_labels)
        return self.sess.run(self.adv, feed_dict={self.model_input: X,
                                                  self.eps: eps,
                                                  self.labels: labels_onehot})


class CarliniAttack:
    def __init__(self, sess, classifier, ord, confidence=None, **kwargs):
        if np.isinf(ord):
            self.attacker = CarliniLi(sess, classifier, **kwargs)
        else:
            self.attacker = CarliniL2(sess, classifier, confidence=confidence, **kwargs)

    def attack(self, X, target_labels):
        input = X / 255. - 0.5
        labels_onehot = one_hot(target_labels)

        img_attack = self.attacker.attack(input, labels_onehot)
        img_attack = np.clip((img_attack + 0.5) * 255, 0, 255)
        return img_attack


def one_hot(y, nclasses=2):
    y_onehot = np.zeros((len(y), nclasses))
    y_onehot[np.arange(len(y)), y] = 1
    return y_onehot