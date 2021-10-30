import unittest

import torch

from utils.gmm_anomaly_scorer import GMMAnomalyScorer


class TestGmmAnomalyScorerMethods(unittest.TestCase):
    def test_gmm_fit(self):
        a = torch.Tensor([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])
        b = torch.Tensor([(13, 14, 15), (16, 17, 18), (18, 19, 20), (21, 22, 23)])

        gmm = GMMAnomalyScorer()
        gmm = gmm.fit(a, b)
        self.assertIsNotNone(gmm)

    def test_gmm_score_not_fitted_should_raise_exception(self):
        gmm = GMMAnomalyScorer()
        self.assertRaises(ValueError, gmm.score)

    def test_gmm_score_fitted_should_return_score(self):
        a = torch.Tensor([(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])
        b = torch.Tensor([(13, 14, 15), (16, 17, 18), (18, 19, 20), (21, 22, 23)])

        gmm = GMMAnomalyScorer()
        score = gmm.fit(a, b).score()

        self.assertIsNotNone(score)

    def test_gmm_plot_method(self):
        a = torch.Tensor([(1, 2), (4, 5), (7, 8), (10, 11)])
        b = torch.Tensor([(13, 14), (16, 17), (18, 19), (21, 22)])

        gmm = GMMAnomalyScorer()
        plot = gmm.fit(a, b).plot()

        self.assertIsNotNone(plot)
