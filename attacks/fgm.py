from typing import Tuple, Union

import torch
import numpy as np


def fgm(model: torch.nn.Module,
        imgs: np.ndarray,
        epsilon: float,
        target_class: int,
        device: torch.device,
        image_constraints: Tuple[int, int]) -> np.ndarray:
    """ Fast Gradient Method

    Parameters
    ----------
    model : torch.nn.Module
        A torch module that outputs predictions N x K, for N samples
        and K classes
    imgs : np.ndarray or torch.tensor (N x C x H x W)
        The image to be attacked
    epsilon : float
        The norm of the attack
    target_class : int
        The desired target class
    device : torch.device
        The torch device used for the computations
    image_constraints : tuple (min, max)
        The minimum and maximum pixel values

    Returns
    -------
    np.ndarray
        The adversarial image

    """
    imgs = torch.tensor(imgs).float().to(device).requires_grad_(True)
    input = imgs.div(255).unsqueeze(0)

    loss = -model(input)[:, target_class].mean()
    g = torch.autograd.grad(loss, imgs)[0]
    g = g / g.norm()

    adv = imgs - epsilon * g
    adv = torch.clamp(adv, image_constraints[0], image_constraints[1])

    return adv.detach().cpu().numpy()
