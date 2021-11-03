"""A torchaudio comform implementation of LFCC features"""
from typing import Optional

import torch
import torchaudio.functional as F
from torchaudio.transforms import AmplitudeToDB, Spectrogram


# TODO: port to pure torch to remove dependecy on numpy
def _create_lin_filter(sample_rate, n_fft, n_filter, f_min=0.0, f_max=None, dtype=torch.float32):
    """Create linear filter bank.

    Based on librosa implementation (https://gist.github.com/RicherMans/dc1b50dd8043cee5872e0b7584f6202f).
    """
    if f_max is None:
        f_max = float(sample_rate) / 2

    # initialize weights
    n_filter = int(n_filter)
    weights = torch.zeros((n_filter, int(1 + n_fft // 2)), dtype=dtype)

    # center freq of each FFT bin
    fftfreqs = torch.linspace(0,
                              float(sample_rate) / 2,
                              int(1 + n_fft//2))

    # 'Center freqs' of liner bands - uniformly spaced between limits
    linear_f = torch.linspace(f_min, f_max, n_filter + 2)

    fdiff = torch.diff(linear_f)
    ramps = linear_f[..., None] - fftfreqs[..., None, :]

    for i in range(n_filter):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = torch.maximum(torch.zeros(1), torch.minimum(lower, upper))

    return weights


class LFCC(torch.nn.Module):
    """Create the linear-frequency cepstral coefﬁcients (LFCC features) from an audio signal.

    By default, this calculates the LFCC features on the DB-scaled linear scaled spectrogram
    to be consistent with the MFCC implementation.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_lin (int, optional): Number of linear filterbanks. (Default: ``128``)
        n_lfcc (int, optional): Number of lfc coefficients to retain. (Default: ``40``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``None``)
        dct_type (int, optional): type of DCT (discrete cosine transform) to use. (Default: ``2``)
        norm (str, optional): norm to use. (Default: ``'ortho'``)
        log_lf (bool, optional): whether to use log lf-spectrograms instead of db-scaled. (Default: ``False``)
        speckwargs (dict or None, optional): arguments for Spectrogram. (Default: ``None``)
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 n_lin: int = 128,
                 n_lfcc: int = 40,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 dct_type: int = 2,
                 norm: str = 'ortho',
                 log_lf: bool = False,
                 speckwargs: Optional[dict] = None
                 ) -> None:
        super().__init__()

        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError('DCT type not supported: {}'.format(dct_type))

        self.sample_rate = sample_rate
        self.n_lin = n_lin
        self.n_lfcc = n_lfcc
        self.f_min = f_min
        self.f_max = f_max
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = AmplitudeToDB('power', self.top_db)

        if speckwargs is not None:
            self.Spectrogram = Spectrogram(**speckwargs)
        else:
            self.Spectrogram = Spectrogram()

        if self.n_lfcc > self.n_lin:
            raise ValueError(
                'Cannot select more LFCC coefficients than # lin bins')

        filter_mat = _create_lin_filter(
            sample_rate=self.sample_rate,
            n_fft=self.Spectrogram.n_fft,
            n_filter=self.n_lin,
            f_min=self.f_min,
            f_max=self.f_max,
        ).T
        self.register_buffer("filter_mat", filter_mat)

        dct_mat = F.create_dct(
            n_lfcc, self.n_lin, self.norm)
        self.register_buffer('dct_mat', dct_mat)
        self.log_lf = log_lf

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
             waveform (Tensor): Tensor of audio of dimension (..., time).
        Returns:
            Tensor: specgram_lf_db of size (..., ``n_lfcc``, time).
        """
        specgram = self.Spectrogram(waveform)

        # adopted from mel scale
        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        specgram = torch.matmul(specgram.transpose(1, 2), self.filter_mat)
        specgram = specgram.transpose(1, 2)

        # unpack batch
        specgram = specgram.reshape(shape[:-2] + specgram.shape[-2:])

        if self.log_lf:
            log_offset = 1e-6
            specgram = torch.log(specgram + log_offset)
        else:
            specgram = self.amplitude_to_DB(specgram)

        lfcc = torch.matmul(specgram.transpose(-2, -1), self.dct_mat)

        return lfcc.transpose(-2, -1)
