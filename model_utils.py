import numpy as np
import torch
from torch.nn import functional as F


class ToTwoOutputs(torch.nn.Module):
    def __init__(self, threshold):
        super(ToTwoOutputs, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.cat((-x + self.threshold, x - self.threshold), dim=1)


class TorchRBFSVM(torch.nn.Module):
    def __init__(self, svm_model, device):
        super(TorchRBFSVM, self).__init__()

        self.support_vectors = torch.tensor(svm_model.support_vectors_).float().to(device)
        self.coef = torch.tensor(svm_model.dual_coef_).float().to(device)
        self.bias = torch.tensor(svm_model.intercept_).float().to(device)
        self.gamma = svm_model.gamma

    def forward(self, x):
        x_augmented = x.unsqueeze(1)
        n_samples = x.shape[0]
        n_vectors = self.support_vectors.shape[0]

        # Prediction of an SVM with RBF kernel. Note that in sklearn the label
        # y of the support vectors is already incorporated in coef
        diff = (x_augmented - self.support_vectors).view(n_samples * n_vectors, -1)
        squared_norm = (diff ** 2).sum(1).view(n_samples, n_vectors)
        scores = (torch.exp(-squared_norm * self.gamma) * self.coef).sum(1, keepdim=True) + self.bias
        return scores


class TorchLinearSVM(torch.nn.Module):
    def __init__(self, svm_model, device):
        super(TorchLinearSVM, self).__init__()

        self.w = torch.tensor(svm_model.coef_).float().to(device)
        self.b = torch.tensor(svm_model.intercept_).float().to(device)

    def forward(self, x):
        scores = F.linear(x, self.w, self.b)
        return scores


class ModelForAnneal:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_score(self, img):
        with torch.no_grad():
            input = torch.tensor(img[np.newaxis]/255).float().to(self.device)
            output = self.model(input)
            return output.cpu().numpy()
