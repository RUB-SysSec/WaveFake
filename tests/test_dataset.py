"""Test cases for datahandling."""
import unittest

import torch
from dfadetect.datasets import AudioDataset
from dfadetect.utils import find_wav_files

from tests.utils import REAL_PATH, load_real, load_special


class TestAudioDataset(unittest.TestCase):

    def test_loading_audio(self):
        dataset = load_real()

        # found all files
        self.assertEqual(len(dataset), 5)

        # returns sample rate
        self.assertEqual(len(dataset[0]), 2)

    def test_resampling(self):
        new_rate = 24_000
        dataset = load_real(sample_rate=new_rate)

        for _, sample_rate in dataset:
            self.assertEqual(sample_rate, new_rate)

    def test_loading_audio_triming(self):
        # trimmed
        dataset = load_real()

        trim_time = 0.
        for waveform, _ in dataset:
            trim_time += waveform.shape[1]

        # not trimmed
        dataset = load_real(trim=False)

        orig_time = 0.
        for waveform, _ in dataset:
            orig_time += waveform.shape[1]

        self.assertGreater(orig_time, trim_time)

    def test_trimming_entire_file(self):
        dataset = load_special()

        # check that we do not trim entire file
        for waveform, _sr in dataset:
            self.assertGreater(waveform.size()[1], 0)

    def test_phone_call(self):
        dataset = load_special(phone_call=True)

        for _waveform, sr in dataset:
            self.assertEqual(sr, 8_000)

    def test_phone_call_reassigned(self):
        dataset = load_special()

        for _waveform, sr in dataset:
            self.assertEqual(sr, 16_000)

        dataset.phone_call = True

        for _waveform, sr in dataset:
            self.assertEqual(sr, 8_000)

    def test_list_of_paths(self):
        ref = load_real()
        paths = find_wav_files(REAL_PATH)
        from_paths = AudioDataset(paths)

        for (file_1, sr_1), (file_2, sr_2) in zip(ref, from_paths):
            self.assertTrue(torch.allclose(file_1, file_2))
            self.assertEqual(sr_1, sr_2)


if __name__ == "__main__":
    unittest.main()
