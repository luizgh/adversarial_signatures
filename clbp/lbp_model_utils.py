from typing import Callable, List, Sequence
import numpy as np
from sklearn.svm import SVC


def onehot(x, nclass=2):
    result = np.zeros((len(x), nclass))
    result[np.arange(len(x)), x] = 1
    return result


class lbp_model:
    def __init__(self,
                 descriptor: Callable,
                 model: SVC,
                 threshold: float):
        """ Model that extracts features using the "descriptor", then uses
            the "model" to obtain a classification score.

        Parameters
        ----------
        descriptor: Callable
            The function that extracts features from an image
        model: sklearn.svm.SVC
            Classifier with a "decision_function" function, that takes a
            feature vector and outputs a score
        threshold: float
            The decision threshold
        """
        self.model = model
        self.threshold = threshold
        self.descriptor = descriptor

    def bounds(self):
        """ Returns the bounds of each pixel in the image"""
        return [0, 255]

    def predictions(self, img: np.ndarray):
        """ Return the prediction for one image

        Parameters
        ----------
        img: np.ndarray
            Input image

        Returns
        -------
        np.ndarray: 1 x 2
            The one-hot prediction (either (1, 0) or (0, 1))

        """
        features = self.descriptor(img)
        pred = self.model.decision_function(features) >= self.threshold
        return onehot(pred.astype(np.int)).squeeze()

    def predict_score(self, img):
        """ Returns the predicted score for one image

        Parameters
        ----------
        img: np.ndarray
            Input image

        Returns
        -------
        float:
            The score of the image, according to the model

        """
        features = self.descriptor(img)
        pred = self.model.decision_function(features)
        return pred

    def batch_predictions(self, imgs: Sequence[np.ndarray]):
        """ Returns predictions for a list of images

        Parameters
        ----------
        imgs: a sequence (e.g. list) of np.ndarray
            List of N images

        Returns
        -------
        np.ndarray (N x 2)
            The one-hot predictions, for each image in the list

        """
        features = [self.descriptor(img) for img in imgs]
        features = np.concatenate(features)
        pred = self.model.decision_function(features) >= self.threshold
        return onehot(pred.astype(np.int))
