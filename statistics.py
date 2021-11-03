import argparse

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchaudio.functional import detect_pitch_frequency, spectral_centroid
from torchaudio.transforms import Spectrogram

from dfadetect.datasets import AudioDataset

N_FFT = 256
SAMPLE_RATE = 22_050

# Latex font
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def _compute_average_frequency_for_directory(directory: str, early_exit=None, compute_stats=True) -> torch.Tensor:
    dataset = dataset = AudioDataset(
        directory,
        sample_rate=SAMPLE_RATE,
        trim=False,
    )
    dataset_mapped = map(lambda x: x[0].squeeze(0), dataset)

    average_per_file = []
    if compute_stats:
        spectral_centroids = []
        pitches = []
        pitches_std = []

    spec_transform = Spectrogram(n_fft=N_FFT)

    for i, (clip, fs) in enumerate(dataset):
        specgram = spec_transform(clip).squeeze(0)

        avg = torch.mean(specgram, dim=1)
        avg_db = 10. * torch.log(avg + 10e-13)
        average_per_file.append(avg_db)

        if i % 100 == 0:
            print(f"\rProcessed {i:06} files!", end="")

        if i == early_exit:
            break

        if compute_stats:
            # compute spectral centroid
            centroid = spectral_centroid(
                waveform=clip,
                sample_rate=fs,
                # same as Spectrogram above
                pad=0,
                window=torch.hann_window(N_FFT),
                n_fft=N_FFT,
                hop_length=N_FFT // 2,
                win_length=N_FFT,
            )
            spectral_centroids.append(torch.mean(centroid))

            # pitch
            pitch = detect_pitch_frequency(
                clip, fs, freq_low=50, freq_high=500)
            pitches.append(torch.mean(pitch))
            pitches_std.append(torch.std(pitch))

    average_per_file = torch.stack(average_per_file)
    average_per_file = torch.mean(average_per_file, dim=0)

    if compute_stats:
        pitches = torch.stack(pitches)
        pitches_std = torch.stack(pitches_std)
        spectral_centroids = torch.stack(spectral_centroids)

        average_centroids = torch.mean(spectral_centroids)
        average_pitch = torch.mean(pitches)
        std_pitch = torch.mean(pitches_std)

        return average_per_file, average_centroids, average_pitch, std_pitch

    else:
        return average_per_file, None, None, None


def _apply_ax_styling(ax, title, num_freqs, y_min=-150., y_max=40, ylabel="Average Energy (dB)"):
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_ylim(y_min, y_max)

    # convert fftbins to freq.
    freqs = np.fft.fftfreq((num_freqs - 1) * 2, 1 /
                           SAMPLE_RATE)[:num_freqs-1] / 1_000

    ticks = ax.get_xticks()[1:]
    ticklabels = (np.linspace(
        freqs[0], freqs[-1], len(ticks)) + .5).astype(np.int32)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)

    ax.set_xlabel("Frequency (kHz)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)


def plot_barplot(data, title, path):
    fig, ax = plt.subplots()

    num_freqs = data.shape[0]

    ax.bar(x=list(range(num_freqs)), height=data, color="#2D5B68")

    _apply_ax_styling(ax, title, num_freqs)

    fig.tight_layout()
    fig.savefig(path)


def plot_difference(data, title, ref_data, ref_title, path, absolute: bool = False):
    fig, axis = plt.subplots(1, 3, figsize=(20, 5))

    num_freqs = ref_data.shape[0]
    # plot ref
    ax = axis[0]
    ax.bar(x=list(range(num_freqs)), height=ref_data, color="#2D5B68")
    _apply_ax_styling(ax, ref_title, num_freqs)

    # plot differnce
    ax = axis[1]
    diff = data - ref_data

    ax.bar(x=list(range(num_freqs)), height=diff, color="crimson")
    if absolute:
        _apply_ax_styling(
            ax, f"absolute differnce {title} - {ref_title}", num_freqs, y_min=0, y_max=10, ylabel="")
        diff = np.abs(diff)
    else:
        _apply_ax_styling(
            ax, f"Differnce {title} - {ref_title}", num_freqs, y_min=-10, y_max=10, ylabel="")

    # plot data
    ax = axis[2]
    ax.bar(x=list(range(num_freqs)), height=data, color="#2D5B68")
    _apply_ax_styling(ax, title, num_freqs)

    fig.tight_layout()
    fig.savefig(path)


def measure_pitch(path):
    y, sr = librosa.load(path)
    f0_curve = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    avg_f0, std_f0 = np.mean(f0_curve), np.std(f0_curve)
    return avg_f0, std_f0


def main(args):
    reference_data = None
    reference_name = None

    for dataset in args.DATASETS:
        path, name = dataset.split(",")
        print("======================================")
        print(f"Processing {name}!")
        print("======================================")
        average_freq, average_centroid, average_pitch, std_pitch = _compute_average_frequency_for_directory(
            path, args.amount, compute_stats=not args.no_stats)
        print(f"\nPitch: {average_pitch} +/- {std_pitch}!")
        print(f"Average centroid: {average_centroid}!")

        plot_barplot(average_freq, name, f"plots/{name.lower().strip()}.pdf")

        if reference_data is None:
            reference_data = average_freq
            reference_name = name
        else:
            plot_difference(average_freq, name, reference_data, reference_name,
                            f"plots/{name.lower().strip()}_differnce.pdf")
            plot_difference(average_freq, name, reference_data, reference_name,
                            f"plots/{name.lower().strip()}_differnce_absolute.pdf", absolute=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("DATASETS", help="Path to datasets. The first entry is assumed to be the referrence one. Specified as follows <path,name>",
                        type=str, nargs="*")
    parser.add_argument(
        "--amount", "-a", help="Amount of files to concider.", type=int, default=None)
    parser.add_argument(
        "--no-stats", "-s", help="Do not compute stats, only plots.", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
