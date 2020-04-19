import numpy as np
import torch

from attacks.anneal import AdversaryAttackProblem
from attacks.carlini import CarliniWagnerL2
from foolbox.attacks import BoundaryAttack
from foolbox.criteria import TargetClass

def carlini_attack(model, img, target_class, device):
    img_01 = (img / 255).astype(np.float32)
    img_01_device = torch.tensor(img_01[np.newaxis]).to(device)
    y_device = torch.tensor([target_class]).to(device)
    atk = CarliniWagnerL2(image_constraints=(0, 1), num_classes=2,
                          confidence=1, device=device, max_iterations=1000)
    adv = atk.attack(model, img_01_device, y_device, targeted=True)
    adv_img = (adv * 255).cpu().numpy()[0]
    return np.clip(adv_img, 0, 255)


def boundary_attack(model, img, target):
    img_01 = (img / 255).astype(np.float32)
    atk = BoundaryAttack(model, TargetClass(target))

    label = 1-target
    adv = atk(img_01, label, iterations=1000, verbose=False,
              log_every_n_steps=100)
    if adv is not None:
        adv = np.clip(adv * 255, 0, 255)
    return adv


def anneal_attack(model, img, threshold, target):
    if target == 0:
        multiplier = 1
    else:
        multiplier = -1
    optim = AdversaryAttackProblem(img, model,
                                   multiplier=multiplier,
                                   norm_weight=1. / 100,
                                   threshold=threshold,
                                   early_stop=True,
                                   std=0.5)

    optim.steps = 1000
    optim.copy_strategy = 'slice'
    optim.Tmax = 1
    optim.Tmin = 0.001
    optim.updates = 100

    adv_img, e = optim.anneal()

    return np.clip(adv_img, 0, 255)


def get_score(model, img, device):
    input = torch.tensor(img[np.newaxis]).float().div(255).to(device)
    with torch.no_grad():
        score = model(input)
    return score.item()


def rmse(X):
    return np.sqrt(np.mean(np.square(X)))
