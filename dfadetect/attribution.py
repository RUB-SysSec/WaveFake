"""Integrated gradient based attribution"""
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


class ModelWrapper:

    def zero_grad(self):
        raise NotImplementedError()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()


def blur_image(spectrogram: torch.Tensor, sigmas: List[float]) -> np.ndarray:
    """For every sigma construct a Gaussian blurred spectrogram.
    Args:
        spectrogram (torch.Tensor): Base spectrogram.
        sigmas (List[float]): List of differnt sigma level.
    """
    spectrograms = []
    for sigma in sigmas:
        spectrograms.append(ndimage.gaussian_filter(
            spectrogram.numpy(), sigma=[sigma, sigma], mode="constant"))

    return np.array(spectrograms)


def compute_gradients(
        spectrograms: np.ndarray,
        model: ModelWrapper,
        feature_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """Compute the gradient attributions for every image.
    Args:
        images - Images to compute gradients for.
        model - Model to use.
        target_class_idx - Id of the target class.
        preprocess_fn - Function for preprocessing the images (default - ImageNet preprocessing).
        binary_classification - classification is simple binary prediction.
    """
    # TODO: Vectorize once torch.vmap is available
    grads = []
    model.eval()

    for spec in spectrograms:
        # forwards pass
        # switch from HxWxC to CxHxW
        spectrogram = torch.from_numpy(spec).detach().requires_grad_(True)
        spectrogram = spectrogram.float()
        if feature_fn is not None:
            inputs = feature_fn(spectrogram)
            inputs = inputs.view(inputs.shape[-2:]).T
        else:
            inputs = spectrogram

        output = model(inputs)

        # compute grad
        model.zero_grad()
        output.backward()

        gradient = spectrogram.grad.detach().cpu()
        grads.append(gradient)

    grads = torch.stack(grads)
    return grads


def integral_approximation(gradients: torch.Tensor) -> torch.Tensor:
    """Use Riemann trapezoidal to approximate the path integral.
    """
    grads = (gradients[:-1] + gradients[1:]) / 2.
    attribution = torch.mean(grads, axis=0)
    return attribution


def blur_gradients(
    spectrogram: torch.Tensor,
    model: ModelWrapper,
    steps: int,
    max_sigma: int = 50,
    feature_fn: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Calculate (blur) integrated gradients.
    Args:
        spectrogram (torch.Tensor): Spectrogram to attribute
        model (torch.nn.Module): Model to use.
        steps (int): Blurring steps to perform.
        max_sigma (int): Maximum sigma of the Gaussian blur.
    """
    # Setup for IG
    baseline_img = torch.zeros_like(spectrogram)

    sigmas = [float(i)*max_sigma/float(steps) for i in range(0, steps+1)]
    interpolation = blur_image(spectrogram, sigmas)

    path_gradients = compute_gradients(
        spectrograms=interpolation, model=model, feature_fn=feature_fn)

    attribution = integral_approximation(path_gradients)

    return baseline_img, interpolation, attribution
