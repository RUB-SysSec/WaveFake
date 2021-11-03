"""Test cases for GMMEM models."""
import unittest

import torch
from dfadetect.models import GMMEM, GMMDescent
from dfadetect.models.gaussian_mixture_model import (classify_dataset,
                                                     score_batch)


class TestGMMBase(unittest.TestCase):
    k = 3
    max_iter = 100
    test_data = torch.randn(100, 2)
    cov_types = ["full", "diag"]


class TestGMMDescent(TestGMMBase):

    def test_models_run(self):
        """Assert that the model runs."""
        model = GMMDescent(self.k, self.test_data)
        model(self.test_data)

    def test_models_run_cuda(self):
        """Assert that the model runs (cuda)."""
        if torch.cuda.is_available():
            # move model to cuda
            model = GMMDescent(self.k, self.test_data).cuda()
            self.assertTrue("cuda" in model.loc.device.type)
            self.assertTrue("cuda" in model.pi.device.type)
            self.assertTrue("cuda" in model.cov.device.type)

            # fit model
            self.assertTrue("cuda" in model._comp.loc.device.type)
            self.assertTrue(
                "cuda" in model._comp.covariance_matrix.device.type)
            self.assertTrue("cuda" in model._mix.logits.device.type)

            model(self.test_data.cuda())
            model.cpu()
            self.assertTrue("cpu" in model.loc.device.type)
            self.assertTrue("cpu" in model.pi.device.type)
            self.assertTrue("cpu" in model.cov.device.type)

            self.assertTrue("cpu" in model._comp.loc.device.type)
            self.assertTrue(
                "cpu" in model._comp.covariance_matrix.device.type)
            self.assertTrue("cpu" in model._mix.logits.device.type)

    def test_different_covariance_types(self):
        for cov_type in ["full", "diag"]:
            model = GMMDescent(self.k, self.test_data,
                               covariance_type=cov_type)
            model(self.test_data)
            model.cuda()(self.test_data.cuda())

    def test_update(self):
        m = GMMDescent(self.k, self.test_data, covariance_type="diag")
        optimizer = torch.optim.SGD(m.parameters(), lr=1e-3)

        pred = m(self.test_data)
        loss = -pred.mean()  # negative log likelihood

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        before_cov = m._comp.covariance_matrix
        before_diag = m.cov.clone()
        optimizer.step()
        m._build_distributions()
        after_cov = m._comp.covariance_matrix
        after_diag = m.cov

        self.assertFalse(torch.allclose(before_cov, after_cov))
        self.assertFalse(torch.allclose(before_diag, after_diag))


class TestGMMEM(TestGMMBase):

    def test_models_run(self):
        """Assert that the model runs."""
        model = GMMEM(self.k, self.test_data, max_iter=self.max_iter)
        model.fit(self.test_data)
        model(self.test_data)

    def test_models_lower_bound_goes_down(self):
        """Assert that the model runs."""
        for cov in self.cov_types:
            with self.subTest(f"Covariance type: {cov}"):
                model = GMMEM(self.k, self.test_data,
                              covariance_type=cov).fit(self.test_data)

                for i in range(len(model._lower_bounds)-1):
                    # lower bound gets better each iteration
                    self.assertGreater(
                        model._lower_bounds[i+1],
                        model._lower_bounds[i],
                    )

    def test_models_run_cuda(self):
        """Assert that the model runs (cuda)."""
        if torch.cuda.is_available():
            # move model to cuda
            model = GMMEM(self.k, self.test_data,
                          max_iter=self.max_iter).cuda()
            self.assertTrue("cuda" in model.loc.device.type)
            self.assertTrue("cuda" in model.pi.device.type)
            self.assertTrue("cuda" in model.cov.device.type)

            # fit model
            model.fit(self.test_data.cuda())
            self.assertTrue("cuda" in model._comp.loc.device.type)
            self.assertTrue(
                "cuda" in model._comp.covariance_matrix.device.type)
            self.assertTrue("cuda" in model._mix.logits.device.type)

            model(self.test_data.cuda())
            model.cpu()
            self.assertTrue("cpu" in model.loc.device.type)
            self.assertTrue("cpu" in model.pi.device.type)
            self.assertTrue("cpu" in model.cov.device.type)

            self.assertTrue("cpu" in model._comp.loc.device.type)
            self.assertTrue(
                "cpu" in model._comp.covariance_matrix.device.type)
            self.assertTrue("cpu" in model._mix.logits.device.type)

    def test_forward_shape(self):
        """Return correct shape."""
        model = GMMEM(self.k, self.test_data, max_iter=self.max_iter)
        model.fit(self.test_data)
        log_probs = model(self.test_data)
        self.assertEqual(log_probs.shape, torch.Size([100, 1]))

    def test_different_covariance_types(self):
        for cov_type in self.cov_types:
            model = GMMEM(self.k, self.test_data, max_iter=self.max_iter,
                          covariance_type=cov_type)
            test_data = torch.randn(100, 2)
            model.fit(test_data)
            model(test_data)
            model.cuda()(test_data.cuda())

    def test_classify_dataset(self):
        model_one = GMMEM(self.k, self.test_data,
                          max_iter=self.max_iter).fit(self.test_data)
        model_two = GMMEM(self.k + 1, self.test_data,
                          max_iter=self.max_iter).fit(self.test_data)

        test_data = torch.randn(10, 2, 2, 100)

        scores = classify_dataset(
            model_one, model_two, test_data, device="cpu")

    def test_score_batch(self):
        model_one = GMMEM(self.k, self.test_data,
                          max_iter=self.max_iter).fit(self.test_data)
        model_two = GMMEM(self.k + 1, self.test_data,
                          max_iter=self.max_iter).fit(self.test_data)

        test_data = torch.randn(10, 100, 2)

        scores = score_batch(
            model_one, model_two, test_data)

        self.assertEqual(torch.Size([10, 100, 1]), scores.shape)

    def test_multiple_runs(self):
        _model = GMMEM(self.k, self.test_data,
                       max_iter=self.max_iter, training_runs=10).fit(self.test_data)


if __name__ == "__main__":
    unittest.main()
