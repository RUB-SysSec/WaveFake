"""Utils for testing."""
import os

from dfadetect.datasets import AudioDataset

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

REAL_PATH = f"{DIR_PATH}/sample_data/LJSpeech"
FAKE_PATH = f"{DIR_PATH}/sample_data/generated"
SPECIAL_PATH = f"{DIR_PATH}/sample_data/special"


def load_real(*args, **kwargs):
    return AudioDataset(REAL_PATH, *args, **kwargs)


def load_fake(*args, **kwargs):
    return AudioDataset(FAKE_PATH, *args, **kwargs)


def load_special(*args, **kwargs):
    return AudioDataset(SPECIAL_PATH, *args, **kwargs)


def load_combined(*args, **kwargs):
    return (load_real(*args, **kwargs), load_fake(*args, **kwargs))
