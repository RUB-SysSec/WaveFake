import logging
from typing import Type

import numpy as np
import torch
import torch.distributions as D
from dfadetect.datasets import TransformDataset
from torch import nn
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)


class GMMException(Exception):
    """Base class for GMM exceptions."""


class NotFittedException(GMMException):
    """Model not fitted."""

    def __init__(self):
        super().__init__("Model not fitted, please call fit on data before inference!")


class UnsupportedCovarinaceType(GMMException):
    """Methods for covarinace estimation not supported."""

    def __init__(self):
        super().__init__("Selected invalid method for estimating covarinace matrix!")


class UnsupportedInitilazationMethod(GMMException):
    """Methods for initializing Gaussian centers not supported."""

    def __init__(self):
        super().__init__("Selected invalid method for initial Gaussian centers!")


class GMMBase(nn.Module):
    """Base class for Gaussian Mixture Models (GMM).

    Args:
        k (int): The number of mixture components.
        data (torch.Tensor): Data to initialize means from. Must provide more data points than clusters!
        loc (torch.Tensor): The means of the Gaussian distributions.
        cov (torch.Tensor): The covariance matrices of the Gaussian distributions.
        covariance_type (str): Which covariance type to learn? Options: {full, diag}. Default: full.
    """

    def __init__(
            self,
            k: int,
            data: torch.Tensor,
            covariance_type: str = "full",
    ) -> None:
        """Initialize the GMM model.

        Args:
            See class description.
        """
        super().__init__()
        self.k = k
        self.covariance_type = covariance_type

        self._mix: D.distribution = None
        self._comp: D.distribution = None

        self._eps = torch.finfo(torch.float32).eps
        self._fitted = False
        self._initalize(data)

    def forward(self, X):
        """Compute the weighted log probabilities for each sample.

        Args:
            X (torch.Tensor): Data matrix (n_samples, n_features).

        Returns:
            log_prob (torch.Tensor): Log probabilities of each data point in X. (n_sample, )
        """
        if not self._fitted:
            raise NotFittedException()

        weighted_log_prob = self._comp.log_prob(X.unsqueeze(
            1)) + torch.log_softmax(self._mix.logits, dim=-1)
        return torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)

    def load_state_dict(self, *args, **kwargs):
        super().load_state_dict(*args, **kwargs)

        # only loc and cov are stored in the state dict, thus we have to build distributions afterwards
        self._build_distributions()
        self._fitted = True

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self._build_distributions()
        return self

    def cuda(self, *args, **kwargs):
        self = super().cuda(*args, **kwargs)
        self._build_distributions()
        return self

    def cpu(self, *args, **kwargs):
        self = super().cpu(*args, **kwargs)
        self._build_distributions()
        return self

    def _build_distributions(self):
        raise NotImplementedError(
            "_build_distributions must be implemented by subclass!")

    def _initalize(self, data: torch.Tensor):
        # equal prior distribution
        d = data.size(1)
        pi = torch.full(fill_value=(1. / self.k),
                        size=[self.k, ])

        # draw from Gaussian distribution
        loc = torch.randn(self.k, d)
        prob = torch.ones(len(data)) / len(data)
        loc = data[torch.multinomial(prob, num_samples=self.k)]

        # simple covariance matrix
        if self.covariance_type == "full":
            cov = torch.stack([torch.eye(d)
                               for _ in range(self.k)])
        elif self.covariance_type == "diag":
            cov = torch.stack([torch.ones(d)
                               for _ in range(self.k)])

        self._initalize_parameters(pi, loc, cov)
        self._build_distributions()

    def _initalize_parameters(self, pi: torch.Tensor, loc: torch.Tensor, cov: torch.Tensor) -> None:
        raise NotImplementedError()


class GMMEM(GMMBase):
    """A Gaussian Mixture Model.

    The model must be trained by calling the fit function.
    It is implemented as a torch.nn.Module, so it can be easily used for creating adversarial
    examples or gradient based attribution.

    Note, in contrast to sklearn, we do not run full k-mean for initialization.
    Instead we use the initilazation procedure from k-means++.

    Args:
        k (int): The number of mixture components.
        data (torch.Tensor): Data to initialize means from.
        d (int): Size of feature dimension.
        loc (torch.Tensor): The means of the Gaussian distributions.
        cov (torch.Tensor): The covariance matrices of the Gaussian distributions.
        max_iter(int): Maximum amount of EM steps.
        tol (float): The convergence threshold. EM iterations will stop when the
            lower bound average gain is below this threshold.
        covariance_type (str): Which covariance type to learn? Options: {full, diag}. Default: full.
        training_runs (int): Amount of retraining the network, returns the network with the highest lower bound.
    """

    def __init__(
            self,
            k: int,
            data: torch.Tensor,
            max_iter: int = 10,
            tol: float = 1e-3,
            covariance_type: str = "full",
            training_runs: int = 1,
    ) -> None:
        """Initialize the GMM model.

        Args:
            See class description.
        """
        super().__init__(k, data, covariance_type)
        self.tol = tol
        self.max_iter = max_iter
        self.training_runs = training_runs
        self._lower_bounds = []

    def fit(self, data):
        """Fit the model using the EM-Algorithm.

        Args:
            data (torch.Tensor): Data matrix (n_samples, n_features).

        Returns:
            self (GMM): The fitted instance.
        """
        self._fitted = True

        best_params = (self.pi.clone(),  self.loc.clone(), self.cov.clone())
        best_lower_bound = None

        LOGGER.debug(f"Starting training...")
        for i in range(1, self.training_runs + 1):
            # reinitalize model each run
            self._initalize(data)

            LOGGER.debug(f"Starting training run {i:03}...")
            counter = 0
            prev_lower_bound = None

            while True:
                prev_loc = self.loc.clone()
                prev_cov = self.cov.clone()

                counter += 1

                # compute responsibilities
                log_resp = self._compute_responsibilites(data)

                # compute helper term Nk
                resp = log_resp.exp()
                Nk = torch.sum(resp, dim=0)  # sum over all points

                # update means
                self.loc = torch.matmul(resp.T, data)
                self.loc = self.loc / (Nk.unsqueeze(1) + self._eps)

                # update cov
                if self.covariance_type == "full":
                    cov = self._update_covariance_full(data, resp, Nk)
                elif self.covariance_type == "diag":
                    cov = self._update_covariance_diag(data, resp, Nk)
                else:
                    raise UnsupportedCovarinaceType()

                self.cov = cov

                self.pi = Nk / len(data)

                # update distributions
                self._build_distributions()

                lower_bound = self.forward(data).mean()
                if prev_lower_bound is None:
                    prev_lower_bound = lower_bound
                    continue

                self._lower_bounds.append(lower_bound)
                change = lower_bound - prev_lower_bound
                if torch.abs(change) < self.tol or counter >= self.max_iter:
                    break

                prev_lower_bound = lower_bound

                LOGGER.debug(
                    f"Finished {counter: 3} iteration(s): Current lower bound: {lower_bound}!")

            LOGGER.debug(
                f"Finished run {i:04} after {counter} iterations (maximum: {self.max_iter})!")

            if best_lower_bound is None or lower_bound > best_lower_bound:
                best_lower_bound = lower_bound
                best_params = (self.pi.clone(),
                               self.loc.clone(), self.cov.clone())

        # restore best model
        self.pi, self.loc, self.cov = best_params
        self._build_distributions()

        return self

    def _initalize_parameters(self, pi, loc, cov):
        self.register_buffer("pi", pi)
        self.register_buffer("loc", loc)
        self.register_buffer("cov", cov)

    def _compute_responsibilites(self, data):
        weighted_log_prob = self._comp.log_prob(data.unsqueeze(
            1)) + torch.log_softmax(self._mix.logits, dim=-1)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)

        return weighted_log_prob - log_prob_norm

    def _update_covariance_full(self, data, resp, Nk):
        cov = torch.zeros_like(self.cov)

        # TODO: updateme when torch.vmap is a thing
        for i in range(self.k):
            respk = resp[:, i]
            nk = Nk[i]

            # update covariances
            diff = data - self.loc[i]
            cov[i] = torch.matmul(respk.unsqueeze(
                0) * diff.T, diff) / (nk + self._eps)

        return cov

    def _update_covariance_diag(self, data, resp, Nk):
        # calculate diagonal vectors
        avg_X2 = torch.matmul(resp.T, data * data)
        avg_X2 = avg_X2 / (Nk.unsqueeze(1) + self._eps)
        avg_means2 = self.loc ** 2
        avg_X_means = self.loc * \
            torch.matmul(resp.T, data) / (Nk.unsqueeze(1) + self._eps)
        cov_diags = avg_X2 - 2 * avg_X_means + avg_means2

        return cov_diags

    def _build_distributions(self):
        # create mutlivariate gaussian
        if self.covariance_type == "full":
            cov = self.cov
            self._comp = D.MultivariateNormal(self.loc, cov)
        elif self.covariance_type == "diag":
            cov = torch.stack([torch.diag(c) for c in self.cov])
            self._comp = D.MultivariateNormal(self.loc, cov)
        else:
            raise UnsupportedCovarinaceType()

        # create mixing weights
        self._mix = D.Categorical(self.pi)


class GMMDescent(GMMBase):
    """A Gaussian Mixture Model.

    The model is designed to be trained by gradient descent, minimizing the negative log likelihood of the data.

    Args:
        k (int): The number of mixture components.
        data (torch.Tensor): Data to initialize means from.
        loc (torch.Tensor): The means of the Gaussian distributions.
        cov (torch.Tensor): The covariance matrices of the Gaussian distributions.
        max_iter(int): Maximum amount of EM steps.
        tol (float): The convergence threshold. EM iterations will stop when the
            lower bound average gain is below this threshold.
        covariance_type (str): Which covariance type to learn? Options: {full, diag}. Default: full.
    """

    def __init__(self, k: int, data: torch.Tensor, covariance_type: str = "full"):
        """Initialize the GMM model.

        Args:
            See class description.
        """
        super().__init__(k, data, covariance_type)

    def _initalize_parameters(self, pi: torch.Tensor, loc: torch.Tensor, cov: torch.Tensor):
        # resize and copy inplace
        self.pi = torch.nn.Parameter(pi)
        self.loc = torch.nn.Parameter(loc)

        # keep diagonal in log space
        if self.covariance_type == "diag":
            # obtain cholesky decomposition of the diagonal
            # convert to diag matrix, obtain decomposition, only keep diagonal
            before = cov.shape
            cov = torch.stack([torch.diagonal(torch.cholesky(torch.diag(cov[i])))
                               for i in range(self.k)])

            assert cov.shape == before

            # keep in log space
            cov = torch.log(cov)
        self.cov = torch.nn.Parameter(cov)

        self._fitted = True

    def _build_distributions(self):
        # create mutlivariate gaussian
        if self.covariance_type == "full":
            cov = self.cov
            self._comp = D.MultivariateNormal(self.loc, cov)
        elif self.covariance_type == "diag":
            # since we keep diagonal in log space
            cov = torch.stack([torch.diag(torch.exp(c)) for c in self.cov])

            # we use tril matrix
            self._comp = D.MultivariateNormal(self.loc, scale_tril=cov)
        else:
            raise UnsupportedCovarinaceType()

        # create mixing weights
        self._mix = D.Categorical(F.softmax(self.pi, dim=0))


def score(real_model: GMMBase, fake_model: GMMBase, data: torch.Tensor) -> torch.Tensor:
    """Score a data point by the log likelihood ratio.

    Args:
        real_model (GMMBase): Model fitted to real data.
        fake_model (GMMBase): Model fitted to fake data.
        data (torch.Tensor): Data point to score.

    Returns:
        log-likelihood ratio (torch.Tensor).
    """
    with torch.no_grad():
        data = data.view(data.shape[-2:])
        return real_model(data).mean() - fake_model(data).mean()


def score_batch(real_model: GMMBase, fake_model: GMMBase, data: torch.Tensor) -> torch.Tensor:
    """Score a batch of data points by the log likelihood ratio.

    Args:
        real_model (GMMBase): Model fitted to real data.
        fake_model (GMMBase): Model fitted to fake data.
        data (torch.Tensor): Data points to score.

    Returns:
        log-likelihood ratio (torch.Tensor).
    """
    batch_dim, n_points, feat_d = data.shape
    # unbatch
    data = data.reshape([batch_dim*n_points, feat_d])
    with torch.no_grad():
        scores = real_model(data) - fake_model(data)
        scores = scores.mean(dim=-1)  # mean over all components

    return scores.reshape([batch_dim, n_points, 1])


def classify_dataset(
        real_model: GMMBase,
        fake_model: GMMBase,
        data: TransformDataset,
        device: str,
) -> torch.Tensor:
    """Score an entire data set.

    Args:
        real_model (GMMBase): Model fitted to real data.
        fake_model (GMMBase): Model fitted to fake data.
        data (torch.Tensor): Data set to be scored.
        device (str): Device to use.

    Returns:
        log-likelihood ratio (torch.Tensor).
    """
    scores = torch.zeros(len(data))
    real_model = real_model.to(device)
    fake_model = fake_model.to(device)
    for i, (wav, _) in enumerate(data):
        wav = torch.transpose(wav, -2, -1).to(device)
        scores[i] = score(real_model, fake_model, wav)

    return scores


def flatten_dataset(
        dataset: TransformDataset,
        device: str,
        amount: int = None,
) -> torch.Tensor:
    """Flatten a dataset for GMM-EM training.

    Note, that we additionally switch time and freq dimensions.

    Args:
        dataset (TransformDataset): The data set to flatten.
        device (str): The device to use.
        amount (int): If supplied, only convert this amount of files.

    Returns:
        The flattened data (torch.Tensor).
    """
    new_data = []
    for i, (wav, _) in enumerate(dataset):
        wav = torch.transpose(wav.squeeze(0), -2, -1).to(device)
        new_data.append(wav)
        if amount is not None and i >= amount:
            break
    new_data = torch.cat(new_data).to(device)
    return new_data


def load_model(
        dataset: torch.Tensor,
        model_path: str,
        device: str,
        clusters: int,
        em: bool,
        covariance_type: str = "diag",
) -> GMMBase:
    if em:
        model_fn: Type[GMMBase] = GMMEM
    else:
        model_fn = GMMDescent

    # get overwritten anyway
    data = flatten_dataset(dataset, device, 10)
    model = model_fn(clusters, data, covariance_type=covariance_type)
    model.load_state_dict(torch.load(model_path))
    return model
