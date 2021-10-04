from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import unittest

import matplotlib.pyplot as plt
import matplotlib.testing.compare
import numpy as np
import scipy.stats as sts
from sklearn.gaussian_process import GaussianProcessRegressor as gpr

from sensitivity_methods import sensitivity
from uq_methods import plots


class TestPlots(unittest.TestCase):
    PRODUCED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_plots_produced'))
    FIDUCIAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_plots_fiducials'))
    tol = 1e-5

    def setUp(self):
        self.X = np.array([[0.46123442, 0.91202244, 0.92876321],
                           [0.06385122, 0.08942266, 0.44186744],
                           [0.68961511, 0.52403073, 0.56833564],
                           [0.85123686, 0.77737199, 0.52990221],
                           [0.28834075, 0.01743994, 0.36242905],
                           [0.54587868, 0.41817851, 0.72421346],
                           [0.25357849, 0.22729639, 0.5369119],
                           [0.15648554, 0.3817992, 0.08481071],
                           [0.06034016, 0.89202591, 0.68325111],
                           [0.47304368, 0.85874444, 0.28670653],
                           [0.13013734, 0.68430436, 0.38675013],
                           [0.42515756, 0.90183769, 0.834895],
                           [0.43438549, 0.56288698, 0.65607153],
                           [0.39890529, 0.66302321, 0.72165207],
                           [0.72526413, 0.30656396, 0.98994899],
                           [0.50290481, 0.02484515, 0.05371448],
                           [0.24154745, 0.19699787, 0.08694096],
                           [0.86792345, 0.73439781, 0.94549644],
                           [0.66545947, 0.05688571, 0.34595029],
                           [0.95990758, 0.05884415, 0.46791467],
                           [0.86474237, 0.79569641, 0.85598485],
                           [0.63037856, 0.10142019, 0.61359389],
                           [0.1390893, 0.1522652, 0.62634887],
                           [0.33668492, 0.36119699, 0.58089201],
                           [0.29715877, 0.73680818, 0.20449131],
                           [0.55683808, 0.49606268, 0.30689581],
                           [0.61112437, 0.45046932, 0.72687226],
                           [0.82401322, 0.0999469, 0.09535599],
                           [0.76943412, 0.13181057, 0.81715459],
                           [0.6067913, 0.52855681, 0.73084772],
                           [0.77817408, 0.23410479, 0.77393847],
                           [0.31784351, 0.55372617, 0.71227582],
                           [0.46758069, 0.35570418, 0.31543622],
                           [0.06989688, 0.82159477, 0.3253991],
                           [0.30579717, 0.56461504, 0.07109011],
                           [0.50532271, 0.57142318, 0.59031356],
                           [0.11022146, 0.1806901, 0.20185294],
                           [0.37793607, 0.45846717, 0.40057469],
                           [0.891715, 0.51578042, 0.05885039],
                           [0.5729882, 0.29217043, 0.12501581],
                           [0.90126537, 0.76969839, 0.52675768],
                           [0.00216229, 0.14707118, 0.9368788],
                           [0.16484123, 0.55441898, 0.83753256],
                           [0.56469525, 0.38370204, 0.65722823],
                           [0.8687188, 0.66432443, 0.67008946],
                           [0.48001072, 0.50522088, 0.13284311],
                           [0.91704432, 0.99687862, 0.65933211],
                           [0.75265367, 0.11150535, 0.9612883],
                           [0.39109998, 0.23905451, 0.6992524],
                           [0.00245559, 0.07515066, 0.7427796],
                           [0.43617553, 0.81086171, 0.76694599],
                           [0.05498618, 0.68288433, 0.05098977],
                           [0.92852732, 0.92922038, 0.07499123],
                           [0.3563414, 0.0369639, 0.80971561],
                           [0.31104242, 0.26773013, 0.24337643],
                           [0.40656828, 0.84523629, 0.92413601],
                           [0.2621117, 0.13541767, 0.13898699],
                           [0.78952943, 0.27979129, 0.36594954],
                           [0.96398771, 0.39427822, 0.42041622],
                           [0.06170219, 0.39562485, 0.0390669],
                           [0.07484891, 0.44352503, 0.86574964],
                           [0.02119805, 0.08114133, 0.66240878],
                           [0.62832535, 0.74553018, 0.33435648],
                           [0.27253955, 0.05851183, 0.9477553],
                           [0.17621574, 0.48891392, 0.08004835],
                           [0.05899438, 0.49678554, 0.39423793],
                           [0.13625638, 0.90123555, 0.99555211],
                           [0.82430987, 0.06799042, 0.98713305],
                           [0.23391724, 0.32835972, 0.11899672],
                           [0.26675385, 0.49923745, 0.5294856],
                           [0.50285101, 0.75814327, 0.30801608],
                           [0.97083411, 0.25323657, 0.91860817],
                           [0.50567205, 0.07012236, 0.12421462],
                           [0.58163984, 0.34303427, 0.36467924],
                           [0.62053834, 0.542813, 0.77542096],
                           [0.04564984, 0.57157144, 0.2524628],
                           [0.98461689, 0.06922946, 0.61206824],
                           [0.18133769, 0.85259098, 0.2208197],
                           [0.02360491, 0.58486804, 0.88898217],
                           [0.24356099, 0.78698977, 0.19156109],
                           [0.3374873, 0.3931525, 0.34168161],
                           [0.04891735, 0.06757889, 0.76139633],
                           [0.19553807, 0.02900628, 0.58441379],
                           [0.08725175, 0.47520548, 0.3877658],
                           [0.72472161, 0.462946, 0.39650086],
                           [0.2661204, 0.82420122, 0.6588341],
                           [0.69032023, 0.53098725, 0.39433453],
                           [0.11943751, 0.74554536, 0.87115827],
                           [0.18756975, 0.83759763, 0.44177224],
                           [0.21552329, 0.82555553, 0.85337084],
                           [0.59029845, 0.40985213, 0.86169482],
                           [0.22949626, 0.30654941, 0.28231961],
                           [0.17845353, 0.94908186, 0.56501311],
                           [0.91970551, 0.04106241, 0.11949207],
                           [0.7979433, 0.50880488, 0.81055288],
                           [0.81982103, 0.36466048, 0.13310552],
                           [0.37220176, 0.99673639, 0.39217999],
                           [0.71401306, 0.82261441, 0.79515913],
                           [0.11756912, 0.45101294, 0.76186856],
                           [0.93985828, 0.92252428, 0.12734155]])

        self.Y = np.stack([self.X.sum(axis=1) ** 2, np.sin(self.X.sum(axis=1))]).T

        self.input_names = ['input_A', 'input_B', 'input_C']
        self.output_names = ['output_A', 'output_B']
        self.ranges = [[0, 1]] * 3

        self.surrogate_model = gpr().fit(self.X, self.Y)

        np.random.seed(20200221)

    def tearDown(self):
        if os.path.isfile('test_model'):
            os.remove('test_model')

    def test_likelihood_plot(self):
        prior = sts.beta(1, 1).rvs(10000)
        posterior = sts.beta(13, 7).rvs(10000)

        fig, ax = plt.subplots(1, 1)

        plots.likelihood_plot(ax, prior, post_points=posterior, bins=50, conf_level=95)

        plt.savefig(self.PRODUCED_DIR + os.sep + 'likelihood_plot_01.png')
        plt.close()

        points = np.linspace(0, 1, 10000)
        weights = sts.beta(13, 7).pdf(points)

        fig, ax = plt.subplots(1, 1)

        plots.likelihood_plot(ax, prior_points=points, post_weights=weights, exp_value=13. / (13. + 7.), bins=50,
                              conf_level=95, density=False)

        plt.savefig(self.PRODUCED_DIR + os.sep + 'likelihood_plot_02.png')
        plt.close()

        self.assertIsNone(
            matplotlib.testing.compare.compare_images(self.FIDUCIAL_DIR + os.sep + 'likelihood_plot_01.png',
                                                      self.PRODUCED_DIR + os.sep + 'likelihood_plot_01.png',
                                                      tol=self.tol))

        self.assertIsNone(
            matplotlib.testing.compare.compare_images(self.FIDUCIAL_DIR + os.sep + 'likelihood_plot_02.png',
                                                      self.PRODUCED_DIR + os.sep + 'likelihood_plot_02.png',
                                                      tol=self.tol))

    def test_box_plot(self):
        prior = sts.beta(1, 1).rvs(10000)
        posterior = sts.beta(13, 7).rvs(10000)

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(10, 10)
        plots.box_plot(ax, prior, posterior_preds=posterior, num_bins=50, exp_obs=13. / (13. + 7.), exp_std=.1)
        plt.savefig(self.PRODUCED_DIR + os.sep + 'box_plot_01.png')
        plt.close()

        points = np.linspace(0, 1, 10000)
        weights = sts.beta(13, 7).pdf(points)

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(10, 10)
        plots.box_plot(ax, prior_preds=points, posterior_wts=weights, num_bins=50, exp_obs=13. / (13. + 7.), exp_std=.1)

        plt.savefig(self.PRODUCED_DIR + os.sep + 'box_plot_02.png')
        plt.close()

        self.assertIsNone(
            matplotlib.testing.compare.compare_images(self.FIDUCIAL_DIR + os.sep + 'box_plot_01.png',
                                                      self.PRODUCED_DIR + os.sep + 'box_plot_01.png',
                                                      tol=self.tol))

        self.assertIsNone(
            matplotlib.testing.compare.compare_images(self.FIDUCIAL_DIR + os.sep + 'box_plot_02.png',
                                                      self.PRODUCED_DIR + os.sep + 'box_plot_02.png',
                                                      tol=self.tol))

    def test_contour_plot(self):
        x_dist = sts.beta(3, 4).rvs(10000)
        y_dist = sts.beta(13, 7).rvs(10000)

        fig, ax = plt.subplots(1, 1)

        plots.contour_plot(ax, x_dist, y_dist, bins=15)
        plt.savefig(self.PRODUCED_DIR + os.sep + 'contour_plot_01.png')
        plt.close()

        x_weighted = sts.uniform(0, 1).rvs(10000)
        y_weighted = sts.uniform(0, 1).rvs(10000)

        weights = sts.beta(3, 4).pdf(x_weighted) * sts.beta(13, 7).pdf(y_weighted)

        fig, ax = plt.subplots(1, 1)

        plots.contour_plot(ax, x_weighted, y_weighted, bins=15, weights=weights)
        plt.savefig(self.PRODUCED_DIR + os.sep + 'contour_plot_02.png')
        plt.close()

        self.assertIsNone(
            matplotlib.testing.compare.compare_images(self.FIDUCIAL_DIR + os.sep + 'contour_plot_01.png',
                                                      self.PRODUCED_DIR + os.sep + 'contour_plot_01.png',
                                                      tol=self.tol))

        self.assertIsNone(
            matplotlib.testing.compare.compare_images(self.FIDUCIAL_DIR + os.sep + 'contour_plot_02.png',
                                                      self.PRODUCED_DIR + os.sep + 'contour_plot_02.png',
                                                      tol=self.tol))

    def test_scatter_plot(self):
        x_dist = sts.beta(3, 4).rvs(10000)
        y_dist = sts.beta(13, 7).rvs(10000)

        fig, ax = plt.subplots(1, 1)

        plots.scatter_plot(ax, x_dist, y_dist, num_points=500)
        plt.savefig(self.PRODUCED_DIR + os.sep + 'scatter_plot_01.png')
        plt.close()

        x_weighted = sts.uniform(0, 1).rvs(10000)
        y_weighted = sts.uniform(0, 1).rvs(10000)

        weights = sts.beta(3, 4).pdf(x_weighted) * sts.beta(13, 7).pdf(y_weighted)

        fig, ax = plt.subplots(1, 1)

        plots.scatter_plot(ax, x_weighted, y_weighted, num_points=500, weights=weights)
        plt.savefig(self.PRODUCED_DIR + os.sep + 'scatter_plot_02.png')
        plt.close()

        self.assertIsNone(
            matplotlib.testing.compare.compare_images(self.FIDUCIAL_DIR + os.sep + 'scatter_plot_01.png',
                                                      self.PRODUCED_DIR + os.sep + 'scatter_plot_01.png',
                                                      tol=self.tol))

        self.assertIsNone(
            matplotlib.testing.compare.compare_images(self.FIDUCIAL_DIR + os.sep + 'scatter_plot_02.png',
                                                      self.PRODUCED_DIR + os.sep + 'scatter_plot_02.png',
                                                      tol=self.tol))

    @unittest.expectedFailure
    def test_sensitivity_plot(self):
        fig, ax = plt.subplots(3, 2)
        fig.set_size_inches(10, 10)
        sensitivity.sensitivity_plot(ax, self.surrogate_model, feature_names=self.input_names,
                                     feature_ranges=self.ranges, response_names=self.output_names,
                                     num_plot_points=100, num_seed_points=5)
        plt.savefig(self.PRODUCED_DIR + os.sep + 'sensitivity_plot_01.png', )
        plt.close()

        self.assertIsNone(
            matplotlib.testing.compare.compare_images(self.FIDUCIAL_DIR + os.sep + 'sensitivity_plot_01.png',
                                                      self.PRODUCED_DIR + os.sep + 'sensitivity_plot_01.png',
                                                      tol=self.tol))

    @unittest.expectedFailure
    def test_variance_network_plot(self):
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(15, 5)
        sensitivity.pce_network_plot(ax[np.newaxis, :], feature_data=self.X, response_data=self.Y,
                                     feature_names=self.input_names, response_names=self.output_names,
                                     feature_ranges=self.ranges)
        plt.savefig(self.PRODUCED_DIR + os.sep + 'variance_network_plot_01.png')
        plt.close()

        self.assertIsNone(
            matplotlib.testing.compare.compare_images(self.FIDUCIAL_DIR + os.sep + 'variance_network_plot_01.png',
                                                      self.PRODUCED_DIR + os.sep + 'variance_network_plot_01.png',
                                                      tol=self.tol))

    @unittest.skipIf(sys.version_info[0] < 3, "Not supported for Python 2")
    @unittest.expectedFailure
    def test_rank_order_plot(self):
        fig, ax = plt.subplots(1, 1)
        sensitivity.mutual_info_rank_plot(ax, feature_data=self.X, response_data=self.Y,
                                          feature_names=self.input_names, response_names=self.output_names)
        plt.savefig(self.PRODUCED_DIR + os.sep + 'rank_order_plot_01.png')
        plt.close()

        self.assertIsNone(
            matplotlib.testing.compare.compare_images(self.FIDUCIAL_DIR + os.sep + 'rank_order_plot_01.png',
                                                      self.PRODUCED_DIR + os.sep + 'rank_order_plot_01.png',
                                                      tol=self.tol))


if __name__ == '__main__':
    unittest.main()
