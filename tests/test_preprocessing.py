import unittest

import torch
from dfadetect.datasets import TransformDataset, double_delta
from dfadetect.lfcc import LFCC

from tests.utils import load_real


class PreprocessingTest(unittest.TestCase):
    def test_transformation_dataset(self):
        lfcc = TransformDataset(dataset=load_real(),
                                transformation=LFCC, needs_sample_rate=True)

        self.assertEqual(len(lfcc), 5)
        for lfcc_feat, _ in lfcc:
            self.assertEqual(lfcc_feat.shape[:-1], torch.Size([1, 40]))

    def test_double_delta(self):
        lfcc = TransformDataset(dataset=load_real(),
                                transformation=LFCC, needs_sample_rate=True)
        dd = double_delta(dataset=lfcc)

        self.assertEqual(len(dd), 5)
        for delta_feat, _ in dd:
            self.assertEqual(delta_feat.shape[:-1], torch.Size([1, 40*3]))

        for (delta_feat, _), (lfcc_feat, _) in zip(dd, lfcc):
            self.assertEqual(delta_feat.shape[-1], lfcc_feat.shape[-1])


if __name__ == "__main__":
    unittest.main()
