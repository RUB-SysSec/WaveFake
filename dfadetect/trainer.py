"""A generic training wrapper."""
import logging
from copy import deepcopy
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dfadetect.datasets import TransformDataset

LOGGER = logging.getLogger(__name__)


class Trainer(object):
    """This is a lightweight wrapper for training models with gradient descent.

    Its main function is to store information about the training process.

    Args:
        epochs (int): The amount of training epochs.
        batch_size (int): Amount of audio files to use in one batch.
        device (str): The device to train on (Default 'cpu').
        batch_size (int): The amount of audio files to consider in one batch (Default: 32).
        optimizer_fn (Callable): Function for constructing the optimzer.
        optimizer_kwargs (dict): Kwargs for the optimzer.
    """

    def __init__(self,
                 epochs: int = 20,
                 batch_size: int = 32,
                 device: str = "cpu",
                 optimizer_fn: Callable = torch.optim.Adam,
                 optimizer_kwargs: dict = {"lr": 1e-3},
                 ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []


class GMMTrainer(Trainer):

    # TODO: return TrainingResult
    # TODO: Make generic to work with something else than GMM
    def train(self,
              model: torch.nn.Module,
              dataset: TransformDataset,
              test_len: float = 0.2,
              ) -> torch.nn.Module:
        """Fit the model given the training data.

        Args:
            model (torch.nn.Module): The model to be fitted.
            loss_fn (Callable): A callable implementing the loss function.
            dataset (torch.utils.Dataset): The dataset for fitting.
            test_len (float): The percentage of data to be used for testing.

        Returns:
            The trained model.
        """
        model = model.to(self.device)
        model.train()

        test_len = int(len(dataset) * test_len)
        train_len = len(dataset) - test_len
        lengths = [train_len, test_len]
        train, test = torch.utils.data.random_split(dataset, lengths)

        # We force batch size as one and assume the different frames of an audio file
        # to be one batch
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=1, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=1)

        optimizer = self.optimizer_fn(
            model.parameters(), **self.optimizer_kwargs)
        LOGGER.debug(f"Starting training for {self.epochs} epochs!")
        for epoch in range(1, self.epochs + 1):

            train_epoch_running = True
            train_iter = iter(train_loader)
            i = 0
            while train_epoch_running:
                # construct batch by merging files
                batch = []
                for _ in range(self.batch_size):
                    try:
                        audio_file, _ = next(train_iter)
                        audio_file = audio_file.view(
                            audio_file.shape[-2:]).T
                        batch.append(audio_file)

                    except StopIteration:
                        train_epoch_running = False
                i += 1
                if len(batch) == 0:
                    break
                batch = torch.cat(batch)

                batch = batch.to(self.device)
                pred = model(batch)
                train_loss = - pred.mean()  # negative log likelihood

                # Backpropagation
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                model._build_distributions()
                if i % 1_00 == 0:
                    LOGGER.debug(
                        f"[{epoch:05}] Current batch loss: {train_loss.item(): 4.3f} [{i:05}/{int(len(train)/self.batch_size):05}]")

            test_loss: List[float] = []
            for audio_file, _ in test_loader:
                # unpack double batching (dataset + dataloader) and reorder to (time, feat)
                audio_file = audio_file.view(
                    audio_file.shape[-2:]).T.to(self.device)
                test_loss.append((-model(audio_file).mean()).item())

            test_loss: float = np.mean(test_loss)
            LOGGER.debug(f"[{epoch:05}] Epoch test loss: {test_loss: 3.3f}")
            self.epoch_test_losses.append(test_loss)

        model.eval()
        return model


class GDTrainer(Trainer):
    def train(self,
              dataset: torch.utils.data.Dataset,
              model: torch.nn.Module,
              test_len: float,
              pos_weight: Optional[torch.FloatTensor] = None,
              ):

        test_len = int(len(dataset) * test_len)
        train_len = len(dataset) - test_len
        lengths = [train_len, test_len]
        train, test = torch.utils.data.random_split(dataset, lengths)

        train_loader = DataLoader(
            train, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            test, batch_size=self.batch_size, drop_last=True)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        best_model = None
        best_acc = 0
        for epoch in range(self.epochs):
            running_loss = 0
            num_correct = 0.0
            num_total = 0.0
            model.train()

            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                batch_size = batch_x.size(0)
                num_total += batch_size

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.unsqueeze(1).type(
                    torch.float32).to(self.device)

                batch_out = model(batch_x)
                batch_loss = criterion(batch_out, batch_y)

                batch_pred = (torch.sigmoid(batch_out) + .5).int()
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

                running_loss += (batch_loss.item() * batch_size)

                optim.zero_grad()
                batch_loss.backward()
                optim.step()

            running_loss /= num_total
            train_accuracy = (num_correct/num_total)*100

            num_correct = 0.0
            num_total = 0.0
            model.eval()
            for batch_x, _, batch_y in test_loader:

                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.unsqueeze(1).type(
                    torch.float32).to(self.device)
                batch_out = model(batch_x)

                batch_pred = (torch.sigmoid(batch_out) + .5).int()
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

            test_acc = 100 * (num_correct / num_total)

            if best_model is None or test_acc > best_acc:
                best_acc = test_acc
                best_model = deepcopy(model.state_dict())

            LOGGER.info(
                f"[{epoch:04d}]: {running_loss} - train acc: {train_accuracy} - test_acc: {test_acc}")

        model.load_state_dict(best_model)
        return model
