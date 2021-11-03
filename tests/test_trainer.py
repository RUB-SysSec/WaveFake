import unittest

from dfadetect.datasets import lfcc
from dfadetect.models import GMMDescent
from dfadetect.trainer import Trainer

from tests.utils import load_fake


class TestTrainer(unittest.TestCase):

    def test_training_gmm_descent(self):
        data = load_fake()
        lfcc_data = lfcc(data)

        model = GMMDescent(3, lfcc_data[0][0][0].T, covariance_type="diag")
        trainer = Trainer(5)
        trainer.train(model, lfcc_data, .2)

        for i in range(len(trainer.epoch_test_losses) - 1):
            self.assertGreater(
                trainer.epoch_test_losses[i],
                trainer.epoch_test_losses[i+1],
            )


if __name__ == "__main__":
    unittest.main()
