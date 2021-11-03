"""An example file for using the attribution methods."""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchaudio.functional as F
from torchaudio.transforms import ComputeDeltas, Spectrogram

from dfadetect.attribution import ModelWrapper, blur_gradients
from dfadetect.datasets import AudioDataset, double_delta, lfcc, mfcc
from dfadetect.models.gaussian_mixture_model import load_model
from experiment_config import feature_kwargs

LATEX_FONT = True

if LATEX_FONT:
    # Latex font
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'


def plot_spectrogram(spec, title=None, ylabel='Filter', aspect='auto', xmax=None):
    if LATEX_FONT:
        ylabel = ylabel.replace("_", "\_")

    fig, axs = plt.subplots(1, 1, figsize=(8, 3))

    # ax styling
    axs.set_xlabel('Frames')

    steps = 2
    amount_of_coefficients = spec.shape[0]

    if amount_of_coefficients == 20:
        axs.set_ylabel(ylabel)
        yticks = np.arange(0, amount_of_coefficients, steps)
        axs.set_yticks(yticks)

        # only display every second tick
        axs.set_yticklabels(
            list(map(lambda x: str(x[1]) if x[0] % 2 == 1 else "", enumerate(yticks))))
    elif amount_of_coefficients == 60:
        # lines seperating LFCC, Delta, Double Delta
        axs.axhline(
            y=20, color="slategray")
        axs.axhline(
            y=40, color="slategray")

        # Tick labels
        yticks = [10, 30, 50]
        axs.set_yticks(yticks)
        ylabel = ["Double\nDelta", "Delta", "LFCC"]
        axs.set_yticklabels(ylabel, fontsize=16, fontweight="bold")

        # do not display ticks
        axs.tick_params(axis="y", which="both", length=0., width=0.)
    else:
        raise ValueError("Unsupported size of coefficients")

    amount_of_frames = spec.shape[1]

    # generate color map
    cmap = sns.diverging_palette(220, 20, l=40, s=100, as_cmap=True)
    im = axs.imshow(spec, origin='lower', aspect=aspect,
                    cmap=cmap, vmin=-0.01, vmax=0.01)

    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)

    dir_path = Path("attribution_plots")
    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    fig.tight_layout()
    fig.savefig(f"{dir_path}/{title.lower().replace(' ', '_')}.pdf")


def attribute_and_plot(spectrogram, model_wrapper, steps, title):
    _, _, attribution = blur_gradients(
        spectrogram=spectrogram,
        model=model_wrapper,
        steps=steps,
        max_sigma=100,
    )
    attribution = attribution.T

    plot_spectrogram(
        attribution, title=f"{title} Attribution")
    plot_spectrogram(
        attribution[:20, :], title=f"{title} Coefficients")
    plot_spectrogram(
        attribution[20:40, :], title=f"{title} Delta")
    plot_spectrogram(
        attribution[40:, :], title=f"{title} Double Delta")


def main(args):
    feature_fn = lfcc if "lfcc" in args.REAL_MODEL else mfcc

    # we load the same file three times, so we avoid errors when loading the model
    dataset = double_delta(feature_fn(
        directory_or_audiodataset=AudioDataset(
            [args.FILE, args.FILE, args.FILE]),
        transformkwargs=feature_kwargs("lfcc" in args.REAL_MODEL),
    ))

    real_model = load_model(
        dataset,
        args.REAL_MODEL,
        device="cpu",
        clusters=args.clusters,
        em="em" in args.REAL_MODEL,
    )

    fake_model = load_model(
        dataset,
        args.FAKE_MODEL,
        device="cpu",
        clusters=args.clusters,
        em="em" in args.REAL_MODEL,
    )

    class GMMWrapper(ModelWrapper):
        def __init__(self, real_model, fake_model):
            super().__init__()
            self.real_model = real_model
            self.fake_model = fake_model

        def zero_grad(self):
            self.real_model.zero_grad()
            self.fake_model.zero_grad()

        def forward(self, x):
            return (self.real_model(x) - self.fake_model(x)).mean()

        def eval(self):
            self.real_model.eval()
            self.fake_model.eval()

    spectrogram = dataset[0][0].T.squeeze(-1)

    attribute_and_plot(spectrogram, GMMWrapper(
        real_model, fake_model), args.steps, "Model Attribution")

    class SingleWrapper(ModelWrapper):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def zero_grad(self):
            self.model.zero_grad()

        def forward(self, x):
            return self.model(x).mean()

        def eval(self):
            self.model.eval()

    attribute_and_plot(spectrogram, SingleWrapper(
        real_model), args.steps, "Real Attribution")
    attribute_and_plot(spectrogram, SingleWrapper(
        fake_model), args.steps, "Fake Attribution")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "FILE", help="Audio sample to attribute.", type=str)
    parser.add_argument(
        "REAL_MODEL", help="Real model to attribute.", type=str)
    parser.add_argument(
        "FAKE_MODEL", help="Fake Model to attribute.", type=str)

    default_k = 128
    parser.add_argument(
        "--clusters", "-k", help=f"The amount of clusters to learn (default: {default_k}).", type=int, default=default_k)

    parser.add_argument(
        "--steps", "-m", help="Amount of steps for integrated gradients.", type=int, default=50)
    parser.add_argument(
        "--blur", "-b", help="Compute BlurIG instead.", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
