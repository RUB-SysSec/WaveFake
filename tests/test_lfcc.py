"""Test LFCC features against reference librosa implementation."""
import unittest

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
from dfadetect.lfcc import LFCC, _create_lin_filter

from tests.utils import load_real


def _lin(sr, n_fft, n_filter=128, fmin=0.0, fmax=None, dtype=np.float32):

    if fmax is None:
        fmax = float(sr) / 2
    # Initialize the weights
    n_filter = int(n_filter)
    weights = np.zeros((n_filter, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of liner bands - uniformly spaced between limits
    linear_f = np.linspace(fmin, fmax, n_filter + 2)

    fdiff = np.diff(linear_f)
    ramps = np.subtract.outer(linear_f, fftfreqs)

    for i in range(n_filter):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    return weights


def _linear_spec(y=None,
                 sr=16_000,
                 n_fft=400,
                 hop_length=200,
                 win_length=400,
                 window='hann',
                 center=True,
                 pad_mode='reflect',
                 power=2.0,
                 **kwargs):
    S = np.abs(
        librosa.core.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          window=window,
                          center=center,
                          pad_mode=pad_mode))**power
    filter = _lin(sr=sr, n_fft=n_fft, **kwargs)
    filtered = np.dot(filter, S)
    return filtered


def _lfcc(y=None,
          sr=1600,
          S=None,
          n_lfcc=40,
          dct_type=2,
          norm='ortho',
          **kwargs):
    if S is None:
        S = librosa.power_to_db(_linear_spec(y=y, sr=sr, **kwargs))
    M = scipy.fftpack.dct(S, axis=0, type=dct_type, norm=norm)[:n_lfcc]
    return M


class LFCCTest(unittest.TestCase):
    def test_filter_creation(self):
        for _ in range(100):
            sample_rate, n_fft, n_filter = np.random.randint(
                low=1, high=5, size=(3))
            sample_rate *= 10_000
            n_fft *= 100
            n_filter *= 32

            self.assertTrue(
                np.allclose(
                    _lin(sample_rate, n_fft, n_filter),
                    _create_lin_filter(sample_rate, n_fft, n_filter),
                    atol=1e-04,
                )
            )

    def test_lfcc(self):
        dataset = load_real()

        for waveform, sr in dataset:
            lfcc = LFCC(sample_rate=sr)

            librosa_lfcc = _lfcc(waveform.T.squeeze(1).numpy(), sr)
            torch_lfcc = lfcc(waveform).numpy()

            self.assertTrue(np.allclose(torch_lfcc, librosa_lfcc, atol=1e-03))


if __name__ == "__main__":
    unittest.main()
