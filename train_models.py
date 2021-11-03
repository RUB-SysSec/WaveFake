"""This file is a simple example script training a CNN model on mel spectrograms."""
import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchaudio.functional import compute_deltas

from dfadetect.datasets import lfcc, load_directory_split_train_test, mfcc
from dfadetect.models.gaussian_mixture_model import (GMMEM, GMMDescent,
                                                     flatten_dataset)
from dfadetect.models.raw_net2 import RawNet
from dfadetect.trainer import GDTrainer, GMMTrainer
from dfadetect.utils import set_seed
from experiment_config import RAW_NET_CONFIG, feature_kwargs

LOGGER = logging.getLogger()


def init_logger(log_file):
    LOGGER.setLevel(logging.INFO)

    # create file handler
    fh = logging.FileHandler(log_file)

    # create console handler
    ch = logging.StreamHandler()

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)


def save_model(
        model: torch.nn.Module,
        model_dir: Union[Path, str],
        name: str,
        em: bool = False,
        raw_net: bool = False
) -> None:
    if raw_net:
        model_class = "raw_net"
    else:
        model_class = "em" if em else "gd"
    full_model_dir = Path(f"{model_dir}/{model_class}/{name}")
    if not full_model_dir.exists():
        full_model_dir.mkdir(parents=True)

    torch.save(model.state_dict(),
               f"{full_model_dir}/ckpt.pth")


def train_models(
        real_training_distribution: Union[Path, str],
        fake_training_distributions: List[Union[Path, str]],
        amount_to_use: int,
        feature_fn: Callable,
        feature_kwargs: dict,
        clusters: int,
        batch_size: int,
        epochs: int,
        retraining: int,
        device: str,
        model_dir: Optional[str] = None,
        use_em: bool = False,
        test_size: float = 0.2,
) -> None:
    LOGGER.info("Loading data...")

    real_dataset_train, _ = load_directory_split_train_test(
        real_training_distribution,
        feature_fn,
        feature_kwargs,
        test_size,
        amount_to_use=amount_to_use,
    )

    if use_em:
        LOGGER.info(
            f"Training real model on {min(len(real_dataset_train), 1_000)} audio files.")
        data = flatten_dataset(real_dataset_train, device,
                               1_000)  # we use 1k files max for EM
        real_model = GMMEM(
            clusters, data, covariance_type="diag", max_iter=epochs, training_runs=retraining).to(device)
        real_model = real_model.fit(data)
    else:
        LOGGER.info(
            f"Training real model on {len(real_dataset_train)} audio files.")
        inital_data = flatten_dataset(real_dataset_train, device, 10)
        real_model = GMMDescent(clusters, inital_data,
                                covariance_type="diag").to(device)
        real_model = GMMTrainer(device=device, epochs=epochs, batch_size=batch_size).train(
            real_model, real_dataset_train, .05)
    LOGGER.info("Training real model done!")
    if model_dir is not None:
        save_model(
            model=real_model,
            model_dir=model_dir,
            name="real",
            em=use_em
        )

    # train fake models
    for current in fake_training_distributions:
        LOGGER.info(f"Training {current}")
        fake_dataset_train, _ = load_directory_split_train_test(
            current,
            feature_fn,
            feature_kwargs,
            test_size,
            amount_to_use=amount_to_use,
        )

        if use_em:
            LOGGER.info(
                f"Training fake model on {min(len(fake_dataset_train), 1_000)} audio files.")
            data = flatten_dataset(fake_dataset_train, device, 1_000)
            fake_model = GMMEM(clusters, data, covariance_type="diag", max_iter=epochs, training_runs=retraining).to(
                device)
            fake_model = fake_model.fit(data)
        else:
            LOGGER.info(
                f"Training fake model on {len(fake_dataset_train)} audio files.")
            inital_data = flatten_dataset(fake_dataset_train, device, 10)
            fake_model = GMMDescent(
                clusters, inital_data, covariance_type="diag").to(device)
            fake_model = GMMTrainer(device=device, epochs=epochs, batch_size=batch_size).train(
                fake_model, fake_dataset_train, .05)

        if model_dir is not None:
            save_model(fake_model, model_dir, str(
                current).strip("/").replace("/", "_"), use_em)

        LOGGER.info("Training fake model done!")

    # train leave-one-out
    LOGGER.info("Training out-of-distribution models!")
    for current in fake_training_distributions:
        LOGGER.info(f"Training all but {current}")
        leave_one_out = set(fake_training_distributions) - set([current])
        leave_one_out_datasets = list(map(lambda x: load_directory_split_train_test(
            x,
            feature_fn,
            feature_kwargs,
            test_size,
            amount_to_use=amount_to_use,
        )[0], leave_one_out))

        if use_em:
            LOGGER.info(
                f"Training fake model on 1,000 audio files.")
            data = map(lambda x: flatten_dataset(
                x, device, 1_000), leave_one_out_datasets)

            individual_size = 1_000 // len(leave_one_out_datasets)
            data = torch.cat(list(map(
                lambda x: x[1][x[0]*individual_size: x[0]*individual_size + individual_size], enumerate(data))))
            fake_model = GMMEM(clusters, data, covariance_type="diag", max_iter=epochs, training_runs=retraining).to(
                device)
            fake_model = fake_model.fit(data)
        else:
            data = ConcatDataset(leave_one_out_datasets)
            LOGGER.info(
                f"Training fake model on {len(data)} audio files.")
            inital_data = flatten_dataset(
                leave_one_out_datasets[0], device, 10)

            fake_model = GMMDescent(
                clusters, inital_data, covariance_type="diag").to(device)
            fake_model = GMMTrainer(device=device, epochs=max(1, epochs // len(leave_one_out_datasets)),
                                    batch_size=batch_size).train(fake_model, data, .05)

        if model_dir is not None:
            save_model(fake_model, model_dir,
                       f"all_but_{str(current).strip('/').replace('/', '_')}", use_em)


def train_raw_net(
        real_training_distribution: Union[Path, str],
        fake_training_distributions: List[Union[Path, str]],
        amount_to_use: int,
        batch_size: int,
        epochs: int,
        device: str,
        model_dir: Optional[str] = None,
        test_size: float = 0.2,
) -> None:

    LOGGER.info("Loading data...")

    real_dataset_train, real_dataset_test = load_directory_split_train_test(
        real_training_distribution,
        None,
        None,
        test_size,
        amount_to_use=amount_to_use,
        pad=True,
        label=1,
    )

    # # train fake models
    for current in fake_training_distributions:
        LOGGER.info(f"Training {current}")
        fake_dataset_train, _ = load_directory_split_train_test(
            current,
            None,
            None,
            test_size,
            amount_to_use=amount_to_use,
            pad=True,
            label=0,
        )

        current_model = RawNet(deepcopy(RAW_NET_CONFIG), device).to(device)
        data_train = ConcatDataset([real_dataset_train, fake_dataset_train])
        LOGGER.info(
            f"Training rawnet model on {len(data_train)} audio files.")

        current_model = GDTrainer(
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            optimizer_kwargs={
                "lr": 0.0001,
                "weight_decay": 0.0001,
            }
        ).train(
            dataset=data_train,
            model=current_model,
            test_len=test_size,
        )

        if model_dir is not None:
            save_model(current_model, model_dir, str(
                current).strip("/").replace("/", "_"), raw_net=True)

    # train leave-one-out
    LOGGER.info("Training out-of-distribution models!")
    for current in fake_training_distributions:
        LOGGER.info(f"Training all but {current}")
        leave_one_out = set(fake_training_distributions) - set([current])

        # equal amount of real and fake data
        leave_one_out_datasets = list(map(lambda x: load_directory_split_train_test(
            x,
            None,
            None,
            test_size,
            amount_to_use=amount_to_use,
            pad=True,
            label=0,
        )[0], leave_one_out))

        current_model = RawNet(deepcopy(RAW_NET_CONFIG), device).to(device)
        data_train = ConcatDataset(
            [real_dataset_train, *leave_one_out_datasets])
        LOGGER.info(
            f"Training rawnet model on {len(data_train)} audio files.")

        leave_one_out_size = sum(map(len, leave_one_out_datasets))
        pos_weight = torch.tensor(
            leave_one_out_size / len(real_dataset_train))

        current_model = GDTrainer(
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            optimizer_kwargs={
                "lr": 0.0001,
                "weight_decay": 0.0001,
            }
        ).train(
            dataset=data_train,
            model=current_model,
            test_len=test_size,
            pos_weight=pos_weight,
        )

        if model_dir is not None:
            save_model(current_model, model_dir,
                       f"all_but_{str(current).strip('/').replace('/', '_')}", raw_net=True)

    LOGGER.info("Training single fake models done!")


def main(args):
    # fix all seeds
    set_seed(42)

    if args.use_em:
        init_logger("experiments_EM.log")
    else:
        init_logger("experiments_gd.log")

    if args.raw_net:
        init_logger("experiments_rawnet.log")

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    device = "cuda" if args.cuda else "cpu"
    feature_fn = lfcc if args.lfcc else mfcc

    # parse fake directories
    base_dir = Path(args.FAKE)
    fake_dirs = []
    for path in base_dir.iterdir():
        if path.is_dir():
            if "jsut" in str(path) or "conformer" in str(path):
                continue

            fake_dirs.append(path.absolute())

    if len(fake_dirs) == 0:
        fake_dirs = [base_dir]

    model_dir_path = f"{args.ckpt}"
    model_dir_path += f"/{'lfcc' if args.lfcc else 'mfcc'}"
    model_dir = Path(model_dir_path)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    if args.raw_net:
        train_raw_net(
            real_training_distribution=args.REAL,
            fake_training_distributions=fake_dirs,
            amount_to_use=args.amount if not args.debug else 100,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_dir=model_dir if not args.debug else None,  # do not save debug models
        )

    else:
        train_models(
            real_training_distribution=args.REAL,
            fake_training_distributions=fake_dirs,
            amount_to_use=args.amount if not args.debug else 100,
            feature_fn=feature_fn,
            feature_kwargs=feature_kwargs(args.lfcc),
            clusters=args.clusters,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            retraining=args.retraining,
            use_em=args.use_em,
            model_dir=model_dir if not args.debug else None,  # do not save debug models
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "REAL", help="Directory containing real data.", type=str)
    parser.add_argument(
        "FAKE", help="Directory containing fake data.", type=str)

    default_amount = None
    parser.add_argument(
        "--amount", "-a", help=f"Amount of files to load from each directory (default: {default_amount} - all).", type=int, default=default_amount)

    default_k = 128
    parser.add_argument(
        "--clusters", "-k", help=f"The amount of clusters to learn (default: {default_k}).", type=int, default=default_k)

    default_batch_size = 8
    parser.add_argument(
        "--batch_size", "-b", help=f"Batch size (default: {default_batch_size}).", type=int, default=default_batch_size)

    default_epochs = 5
    parser.add_argument(
        "--epochs", "-e", help=f"Epochs (default: {default_epochs}).", type=int, default=default_epochs)

    default_retraining = 10
    parser.add_argument(
        "--retraining", "-r", help=f"Retraining tries (default: {default_retraining}).", type=int, default=default_retraining)

    default_model_dir = "trained_models"
    parser.add_argument(
        "--ckpt", help=f"Checkpoint directory (default: {default_model_dir}).", type=str, default=default_model_dir)

    parser.add_argument(
        "--use_em", help="Use EM version?", action="store_true")
    parser.add_argument(
        "--raw_net", help="Train raw net version?", action="store_true")
    parser.add_argument(
        "--cuda", "-c", help="Use cuda?", action="store_true")
    parser.add_argument(
        "--lfcc", "-l", help="Use LFCC instead of MFCC?", action="store_true")
    parser.add_argument(
        "--debug", "-d", help="Only use minimal amount of files?", action="store_true")
    parser.add_argument(
        "--verbose", "-v", help="Display debug information?", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
