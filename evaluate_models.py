import argparse
import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, plot_roc_curve, roc_curve
from torch.utils.data import DataLoader

from dfadetect.datasets import (TransformDataset, lfcc,
                                load_directory_split_train_test, mfcc)
from dfadetect.models.gaussian_mixture_model import (GMMBase, classify_dataset,
                                                     load_model)
from dfadetect.models.raw_net2 import RawNet
from dfadetect.utils import set_seed
from experiment_config import RAW_NET_CONFIG, feature_kwargs

DATASET_RE = re.compile("((ljspeech|waveglow).*)\/")

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


def plot_roc(
        fpr: np.ndarray,
        tpr: np.ndarray,
        training_dataset_name: str,
        fake_dataset_name: str,
        path: str,
        lw: int = 2
):
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(
        f'Train: {training_dataset_name}\nEvaluated on {fake_dataset_name}')
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(f"{path}.pdf")
    plt.close(fig)


def calculate_eer(y, y_score) -> Tuple[float, float, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y, -y_score)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return thresh, eer, fpr, tpr


def calculate_eer_for_models(
    real_model: GMMBase,
    fake_model: GMMBase,
    real_dataset_test: TransformDataset,
    fake_dataset_test: TransformDataset,
    training_dataset_name: str,
    fake_dataset_name: str,
    plot_dir_path: str,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    real_scores = classify_dataset(
        real_model,
        fake_model,
        real_dataset_test,
        device
    ).numpy()

    fake_scores = classify_dataset(
        real_model,
        fake_model,
        fake_dataset_test,
        device
    ).numpy()

    # JSUT fake samples are fewer available
    length = min(len(real_scores),  len(fake_scores))
    real_scores = real_scores[:length]
    fake_scores = fake_scores[:length]

    labels = np.concatenate(
        (
            np.zeros(real_scores.shape, dtype=np.int32),
            np.ones(fake_scores.shape, dtype=np.int32)
        )
    )

    thresh, eer, fpr, tpr = calculate_eer(
        y=np.array(labels, dtype=np.int32),
        y_score=np.concatenate((real_scores, fake_scores)),
    )

    fig_path = f"{plot_dir_path}/{training_dataset_name.replace('.', '_').replace('/', '_')}_{fake_dataset_name.replace('.', '_').replace('/', '_')}"
    plot_roc(fpr, tpr, training_dataset_name, fake_dataset_name, fig_path)

    return eer, thresh, fpr, tpr


def _classify_dataset(model: RawNet, dataset: TransformDataset, device: str, batch_size: int = 128) -> torch.Tensor:
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False)
    model.eval()
    result = []
    with torch.no_grad():
        for (batch_x, _, _) in data_loader:
            batch_x = batch_x.to(device)
            result.append(model(batch_x).squeeze(1))

    return torch.hstack(result)


def calculate_eer_for_raw_net(
    model: RawNet,
    real_dataset_test: TransformDataset,
    fake_dataset_test: TransformDataset,
    training_dataset_name: str,
    fake_dataset_name: str,
    plot_dir_path: str,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    real_scores = _classify_dataset(
        model,
        real_dataset_test,
        device
    ).cpu().numpy()

    fake_scores = _classify_dataset(
        model,
        fake_dataset_test,
        device
    ).cpu().numpy()

    # JSUT fake samples are fewer available
    length = min(len(real_scores),  len(fake_scores))
    real_scores = real_scores[:length]
    fake_scores = fake_scores[:length]

    labels = np.concatenate(
        (
            np.zeros(real_scores.shape, dtype=np.int32),
            np.ones(fake_scores.shape, dtype=np.int32)
        )
    )

    thresh, eer, fpr, tpr = calculate_eer(
        y=np.array(labels, dtype=np.int32),
        y_score=np.concatenate((real_scores, fake_scores)),
    )

    fig_path = f"{plot_dir_path}/{training_dataset_name.replace('.', '_').replace('/', '_')}_{fake_dataset_name.replace('.', '_').replace('/', '_')}"
    plot_roc(fpr, tpr, training_dataset_name, fake_dataset_name, fig_path)

    return eer, thresh, fpr, tpr


def evaluate(
        real_model_path: str,
        fake_models_path: dict,
        real_dir: Union[Path, str],
        fake_dirs: List[Union[Path, str]],
        amount_to_use: Optional[int],
        feature_fn: Callable,
        feature_kwargs: dict,
        clusters: int,
        device: str,
        em: bool,
        output_file_name: str,
        test_size: float = 0.2,
        test_dirs: List[Union[Path, str]] = [],
):

    _, real_dataset_test = load_directory_split_train_test(
        real_dir,
        feature_fn,
        feature_kwargs,
        test_size,
        amount_to_use=amount_to_use,
    )

    real_model = load_model(
        real_dataset_test,
        real_model_path,
        device,
        clusters,
        em,
    )

    complete_results = {}

    for fake_dir in map(str, fake_dirs):
        data_set_name = fake_dir[fake_dir.rfind("/") + 1:]
        if not data_set_name in fake_models_path:
            continue

        # create plot dir for training set
        plot_path = Path(
            f"plots/{'em' if em else 'gd'}/{'lfcc' if 'lfcc' in real_model_path else 'mfcc'}/{fake_dir.replace('.', '_').replace('/', '_')}")
        if not plot_path.exists():
            plot_path.mkdir(parents=True)
        plot_path = str(plot_path)

        results = {"training_set": fake_dir}
        LOGGER.info(f"Evaluating {fake_dir}...")
        fake_model_path = fake_models_path[data_set_name]

        _, fake_dataset_test = load_directory_split_train_test(
            fake_dir,
            feature_fn,
            feature_kwargs,
            test_size,
            amount_to_use=amount_to_use,
        )

        fake_model = load_model(
            fake_dataset_test,
            fake_model_path,
            device,
            clusters,
            em,
        )

        LOGGER.info(f"Calculating in-distribution...")

        eer, thresh, fpr, tpr = calculate_eer_for_models(
            real_model,
            fake_model,
            real_dataset_test,
            fake_dataset_test,
            fake_dir,
            fake_dir,
            plot_dir_path=plot_path,
            device=device,
        )
        results["eer"] = str(eer)
        results["thresh"] = str(thresh)
        results["fpr"] = str(list(fpr))
        results["tpr"] = str(list(tpr))

        LOGGER.info(f"{fake_dir}:\n\tEER: {eer} Thresh: {thresh}")
        LOGGER.info(f"Calculating out-distribution...")
        fake_out_results = {}
        for fake_dir_out in map(str, fake_dirs):
            if fake_dir_out == fake_dir:
                continue

            _, fake_out_dataset_test = load_directory_split_train_test(
                fake_dir_out,
                feature_fn,
                feature_kwargs,
                test_size,
                amount_to_use=amount_to_use,
            )

            eer, thresh, fpr, tpr = calculate_eer_for_models(
                real_model,
                fake_model,
                real_dataset_test,
                fake_out_dataset_test,
                fake_dir,
                fake_dir_out,
                plot_dir_path=plot_path,
                device=device,
            )
            LOGGER.info(f"{fake_dir_out}:\n\tEER: {eer} Thresh: {thresh}")
            fake_out_results[fake_dir_out] = {
                "eer": str(eer),
                "thresh": str(thresh),
                "fpr": str(list(fpr)),
                "tpr": str(list(tpr)),
            }

        results["out_distribution"] = fake_out_results

        test_results = {}
        for test_dir in map(str, test_dirs):
            _, test_dataset_test = load_directory_split_train_test(
                test_dir,
                feature_fn,
                feature_kwargs,
                test_size,
                amount_to_use=amount_to_use,
            )

            eer, thresh, fpr, tpr = calculate_eer_for_models(
                real_model=real_model,
                fake_model=fake_model,
                real_dataset_test=real_dataset_test,
                fake_dataset_test=test_dataset_test,
                training_dataset_name=fake_dir,
                fake_dataset_name=test_dir,
                plot_dir_path=plot_path,
                device=device,
            )
            LOGGER.info(f"{test_dir}:\n\tEER: {eer} Thresh: {thresh}")
            test_results[test_dir] = {
                "eer": str(eer),
                "thresh": str(thresh),
                "fpr": str(list(fpr)),
                "tpr": str(list(tpr)),
            }

        results["test"] = test_results

        phone_results = {}
        for test_dir in map(str, test_dirs):
            _, test_dataset_test = load_directory_split_train_test(
                test_dir,
                feature_fn,
                feature_kwargs,
                test_size,
                amount_to_use=amount_to_use,
                phone_call=True,
            )

            eer, thresh, fpr, tpr = calculate_eer_for_models(
                real_model=real_model,
                fake_model=fake_model,
                real_dataset_test=real_dataset_test,
                fake_dataset_test=test_dataset_test,
                training_dataset_name=fake_dir,
                fake_dataset_name=test_dir,
                plot_dir_path=plot_path,
                device=device,
            )
            LOGGER.info(
                f"{test_dir}-Phone Call:\n\tEER: {eer} Thresh: {thresh}")
            phone_results[f"{test_dir}_phone"] = {
                "eer": str(eer),
                "thresh": str(thresh),
                "fpr": str(list(fpr)),
                "tpr": str(list(tpr)),
            }

        results["phone"] = phone_results

        complete_results[fake_dir] = (results)

    with open(f"{output_file_name}.json", "w+") as json_file:
        json.dump(complete_results, json_file, indent=4)


def evaluate_raw_net(
        model_paths: dict,
        real_dir: Union[Path, str],
        fake_dirs: List[Union[Path, str]],
        amount_to_use: Optional[int],
        device: str,
        output_file_name: str,
        test_size: float = 0.2,
        test_dirs: List[Union[Path, str]] = [],
):

    _, real_dataset_test = load_directory_split_train_test(
        real_dir,
        None,
        None,
        test_size,
        amount_to_use=amount_to_use,
        pad=True,
        label=1,
    )

    complete_results = {}

    for fake_dir in map(str, fake_dirs):
        data_set_name = fake_dir[fake_dir.rfind("/") + 1:]
        if data_set_name not in model_paths:
            continue

        # create plot dir for training set
        plot_path = Path(
            f"plots/raw_net/{fake_dir.replace('.', '_').replace('/', '_')}")
        if not plot_path.exists():
            plot_path.mkdir(parents=True)
        plot_path = str(plot_path)

        results = {"training_set": fake_dir}
        LOGGER.info(f"Evaluating {fake_dir}...")
        current_model_path = model_paths[data_set_name]

        _, fake_dataset_test = load_directory_split_train_test(
            fake_dir,
            None,
            None,
            test_size,
            amount_to_use=amount_to_use,
            pad=True,
            label=0,
        )

        current_model = RawNet(deepcopy(RAW_NET_CONFIG), device)
        current_model.load_state_dict(
            torch.load(current_model_path))
        current_model = current_model.to(device)

        LOGGER.info(f"Calculating in-distribution...")

        eer, thresh, fpr, tpr = calculate_eer_for_raw_net(
            current_model,
            real_dataset_test,
            fake_dataset_test,
            fake_dir,
            fake_dir,
            plot_dir_path=plot_path,
            device=device,
        )
        results["eer"] = str(eer)
        results["thresh"] = str(thresh)
        results["fpr"] = str(list(fpr))
        results["tpr"] = str(list(tpr))

        LOGGER.info(f"{fake_dir}:\n\tEER: {eer} Thresh: {thresh}")
        LOGGER.info(f"Calculating out-distribution...")
        fake_out_results = {}
        for fake_dir_out in map(str, fake_dirs):
            if fake_dir_out == fake_dir:
                continue

            _, fake_out_dataset_test = load_directory_split_train_test(
                fake_dir_out,
                None,
                None,
                test_size,
                amount_to_use=amount_to_use,
                pad=True,
                label=0,
            )

            eer, thresh, fpr, tpr = calculate_eer_for_raw_net(
                current_model,
                real_dataset_test,
                fake_out_dataset_test,
                fake_dir,
                fake_dir_out,
                plot_dir_path=plot_path,
                device=device,
            )
            LOGGER.info(f"{fake_dir_out}:\n\tEER: {eer} Thresh: {thresh}")
            fake_out_results[fake_dir_out] = {
                "eer": str(eer),
                "thresh": str(thresh),
                "fpr": str(list(fpr)),
                "tpr": str(list(tpr)),
            }

        results["out_distribution"] = fake_out_results

        test_results = {}
        for test_dir in map(str, test_dirs):
            _, test_dataset_test = load_directory_split_train_test(
                test_dir,
                None,
                None,
                test_size,
                amount_to_use=amount_to_use,
                pad=True,
                label=0,
            )

            eer, thresh, fpr, tpr = calculate_eer_for_raw_net(
                current_model,
                real_dataset_test=real_dataset_test,
                fake_dataset_test=test_dataset_test,
                training_dataset_name=fake_dir,
                fake_dataset_name=test_dir,
                plot_dir_path=plot_path,
                device=device,
            )

            LOGGER.info(f"{test_dir}:\n\tEER: {eer} Thresh: {thresh}")
            test_results[test_dir] = {
                "eer": str(eer),
                "thresh": str(thresh),
                "fpr": str(list(fpr)),
                "tpr": str(list(tpr)),
            }

        results["test"] = test_results

        phone_results = {}
        for test_dir in map(str, test_dirs):
            _, test_dataset_test = load_directory_split_train_test(
                test_dir,
                None,
                None,
                test_size,
                amount_to_use=amount_to_use,
                pad=True,
                label=0,
                phone_call=True,
            )

            eer, thresh, fpr, tpr = calculate_eer_for_raw_net(
                current_model,
                real_dataset_test=real_dataset_test,
                fake_dataset_test=test_dataset_test,
                training_dataset_name=fake_dir,
                fake_dataset_name=test_dir,
                plot_dir_path=plot_path,
                device=device,
            )

            LOGGER.info(
                f"{test_dir}-Phone Call:\n\tEER: {eer} Thresh: {thresh}")
            phone_results[f"{test_dir}_phone"] = {
                "eer": str(eer),
                "thresh": str(thresh),
                "fpr": str(list(fpr)),
                "tpr": str(list(tpr)),
            }

        results["phone"] = phone_results

        complete_results[fake_dir] = (results)

    with open(f"{output_file_name}.json", "w+") as json_file:
        json.dump(complete_results, json_file, indent=4)


def main(args):
    # fix all seeds - this should not actually change anything
    set_seed(42)

    device = "cuda" if args.cuda else "cpu"
    feature_fn = lfcc if "lfcc" in args.MODELS else mfcc

    model_dir = Path(args.MODELS)
    real_model = None
    fake_models = {}
    for path in model_dir.glob("**/*.pth"):
        path = str(path)
        if "real" in path:
            real_model = path
        else:
            data_set_name = DATASET_RE.search(path)[1]
            fake_models[data_set_name] = path

    # parse fake directories
    base_dir = Path(args.FAKE)
    fake_dirs = []
    test_dirs = []
    for path in base_dir.iterdir():
        if path.is_dir():
            if "jsut" in str(path) or "common_voices" in str(path):
                test_dirs.append(path.absolute())

            else:
                fake_dirs.append(path.absolute())

    if len(fake_dirs) == 0:
        fake_dirs = [base_dir]

    if args.raw_net:
        evaluate_raw_net(
            model_paths=fake_models,
            real_dir=args.REAL,
            fake_dirs=fake_dirs,
            test_dirs=test_dirs,
            device=device,
            amount_to_use=args.amount if not args.debug else 100,
            output_file_name=args.output,
        )
    else:
        evaluate(
            real_model_path=real_model,
            fake_models_path=fake_models,
            real_dir=args.REAL,
            fake_dirs=fake_dirs,
            test_dirs=test_dirs,
            device=device,
            em="em" in real_model,
            amount_to_use=args.amount if not args.debug else 100,
            feature_kwargs=feature_kwargs("lfcc" in args.MODELS),
            feature_fn=feature_fn,
            clusters=args.clusters,
            output_file_name=args.output,
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "REAL", help="Directory containing real data.", type=str)
    parser.add_argument(
        "FAKE", help="Directory containing fake data.", type=str)
    parser.add_argument(
        "MODELS", help="Directory containing model checkpoints.", type=str)

    parser.add_argument(
        "--output", "-o", help="Output file name.", type=str, default="results")

    default_k = 128
    parser.add_argument(
        "--clusters", "-k", help=f"The amount of clusters to learn (default: {default_k}).", type=int, default=default_k)

    default_amount = None
    parser.add_argument(
        "--amount", "-a", help=f"Amount of files to load from each directory (default: {default_amount} - all).", type=int, default=default_amount)

    parser.add_argument(
        "--raw_net", "-r", help="RawNet models?", action="store_true")
    parser.add_argument(
        "--debug", "-d", help="Only use minimal amount of files?", action="store_true")
    parser.add_argument(
        "--cuda", "-c", help="Use cuda?", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
