from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

import numpy as np
from uqp.surrogate_model.surrogate_model import SurrogateModel


def tensorflow_availible():
    try:
        import tensorflow
    except ImportError:
        return False
    else:
        return True


class TestSurrogateModels(unittest.TestCase):

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

        self.Y = self.X ** 2

        self.X_star = np.array([[0.60388939, 0.74495308, 0.33493706],
                                [0.40558096, 0.6093988, 0.16514972],
                                [0.71804326, 0.29617944, 0.50891022],
                                [0.69053977, 0.579304, 0.83311891],
                                [0.59050287, 0.06109763, 0.20991066],
                                [0.59746492, 0.99981946, 0.58997147],
                                [0.4111654, 0.03735756, 0.77944008],
                                [0.39124101, 0.17395209, 0.75297837],
                                [0.98742826, 0.78362136, 0.41531116],
                                [0.21523426, 0.81678441, 0.71399373]])

        self.input_names = ['input_A', 'input_B', 'input_C']
        self.output_names = ['output_A', 'output_B', 'output_C']
        self.ranges = [[0, 1]] * 3

    def tearDown(self):
        if os.path.isfile('test_model'):
            os.remove('test_model')

    def test_bayesian_ridge_valid(self):
        model = SurrogateModel(model_type='brr', X_names=self.input_names, Y_names=self.output_names,
                               X_range=self.ranges)

        model.fit(self.X, self.Y)

        model.save('test_model')

        model2 = SurrogateModel.load('test_model')

        pred = model2.predict(self.X_star, return_std=True)

        Y_star_actual = pred[0]
        Y_star_std_actual = pred[1]

        Y_star_expected = [[0.43158394, 0.56274887, 0.16415395],
                           [0.24137249, 0.42719427, -0.00676405],
                           [0.55005738, 0.14055735, 0.34622826],
                           [0.5215334, 0.41757623, 0.66971459],
                           [0.42862527, -0.09198735, 0.04659451],
                           [0.42328907, 0.8114844, 0.41830576],
                           [0.26058284, -0.10351873, 0.62068153],
                           [0.2390059, 0.02578383, 0.59209856],
                           [0.80112185, 0.60615729, 0.24722264],
                           [0.05888063, 0.63528935, 0.54341996]]

        Y_star_std_expected = [[0.07998235, 0.08039285, 0.0763507],
                               [0.07807029, 0.0784711, 0.07452488],
                               [0.07916137, 0.07956771, 0.07556669],
                               [0.08171384, 0.08213314, 0.07800402],
                               [0.07727931, 0.07767611, 0.07376953],
                               [0.08264642, 0.08307044, 0.07889459],
                               [0.07890787, 0.07931293, 0.07532467],
                               [0.07875423, 0.07915851, 0.07517794],
                               [0.08288926, 0.08331451, 0.07912648],
                               [0.08068222, 0.08109628, 0.07701897]]

        np.testing.assert_array_almost_equal(Y_star_actual, Y_star_expected)
        np.testing.assert_array_almost_equal(Y_star_std_actual, Y_star_std_expected)

    def test_gaussian_process_regressor_valid(self):
        model = SurrogateModel(model_type='gpr', X_names=self.input_names, Y_names=self.output_names,
                               X_range=self.ranges)

        model.fit(self.X, self.Y)
        model.save('test_model')

        model2 = SurrogateModel.load('test_model')
        pred = model2.predict(self.X_star, return_std=True)

        Y_star_actual = pred[0]
        Y_star_std_actual = pred[1]

        Y_star_expected = [[0.36467733, 0.55495554, 0.11218544],
                           [0.16448758, 0.37135334, 0.02728771],
                           [0.51557825, 0.08772156, 0.25900691],
                           [0.47685433, 0.33558669, 0.6941037],
                           [0.34869859, 0.0037424, 0.04405257],
                           [0.35691604, 0.99977371, 0.34815954],
                           [0.16908372, 0.001392, 0.60750473],
                           [0.15307005, 0.03027741, 0.56698415],
                           [0.97485058, 0.61397089, 0.17247533],
                           [0.04632909, 0.66713238, 0.50977249]]

        Y_star_std_expected = [[0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.]]

        np.testing.assert_array_almost_equal(Y_star_actual, Y_star_expected)
        np.testing.assert_allclose(Y_star_std_actual, Y_star_std_expected, atol=1e-2)


    @unittest.skipIf(True, 'mars is not availible')
    def test_mars_valid(self):
        model = SurrogateModel(model_type='mars', X_names=self.input_names, Y_names=self.output_names,
                               X_range=self.ranges)

        model.fit(self.X, self.Y)
        model.save('test_model')

        model2 = SurrogateModel.load('test_model')
        pred = model2.predict(self.X_star, return_std=True)

        Y_star_actual = pred[0]
        Y_star_std_actual = pred[1]

        Y_star_expected = [[0.36492418, 0.55359755, 0.11215437],
                           [0.1683431, 0.37409695, 0.02747061],
                           [0.51647208, 0.0880175, 0.25918949],
                           [0.47852748, 0.33962918, 0.69331232],
                           [0.34971094, 0.00393824, 0.04007303],
                           [0.35762303, 0.99065293, 0.349589],
                           [0.17230653, -0.00219649, 0.60659395],
                           [0.15816564, 0.03310123, 0.56737695],
                           [0.96743312, 0.60951423, 0.17462892],
                           [0.04352139, 0.66705996, 0.50960062]]

        np.testing.assert_array_almost_equal(Y_star_actual, Y_star_expected)
        self.assertIsNone(Y_star_std_actual)

    @unittest.skipUnless(tensorflow_availible(), 'test requires tensorflow')
    def test_djinn_valid(self):
        model = SurrogateModel(model_type='djinn', X_names=self.input_names, Y_names=self.output_names,
                               X_range=self.ranges)

        model.fit(self.X, self.Y)
        model.save('test_model')

        model2 = SurrogateModel.load('test_model')
        pred = model2.predict(self.X_star, return_std=True)

        Y_star_actual = pred[0]
        Y_star_std_actual = pred[1]

        Y_star_expected = [[0.36467733, 0.55495554, 0.11218544],
                           [0.16448758, 0.37135334, 0.02728771],
                           [0.51557825, 0.08772156, 0.25900691],
                           [0.47685433, 0.33558669, 0.6941037],
                           [0.34869859, 0.0037424, 0.04405257],
                           [0.35691604, 0.99977371, 0.34815954],
                           [0.16908372, 0.001392, 0.60750473],
                           [0.15307005, 0.03027741, 0.56698415],
                           [0.97485058, 0.61397089, 0.17247533],
                           [0.04632909, 0.66713238, 0.50977249]]

        Y_star_std_expected = [[0.00181254, 0.00181254, 0.00181254],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0.0020183, 0.0020183, 0.0020183]]

        np.testing.assert_array_almost_equal(Y_star_actual, Y_star_expected)
        np.testing.assert_array_almost_equal(Y_star_std_actual, Y_star_std_expected)

    def test_bayesian_gaussian_process_valid(self):
        model = SurrogateModel(model_type='bgp', X_names=self.input_names, Y_names=self.output_names,
                               X_range=self.ranges)

        model.fit(self.X, self.Y)
        model.save('test_model')

        model2 = SurrogateModel.load('test_model')
        pred = model2.predict(self.X_star, return_std=True)

        Y_star_actual = pred[0]
        Y_star_std_actual = pred[1]

        Y_star_expected = [[0.36467733, 0.55495554, 0.11218544],
                           [0.16448758, 0.37135334, 0.02728771],
                           [0.51557825, 0.08772156, 0.25900691],
                           [0.47685433, 0.33558669, 0.6941037],
                           [0.34869859, 0.0037424, 0.04405257],
                           [0.35691604, 0.99977371, 0.34815954],
                           [0.16908372, 0.001392, 0.60750473],
                           [0.15307005, 0.03027741, 0.56698415],
                           [0.97485058, 0.61397089, 0.17247533],
                           [0.04632909, 0.66713238, 0.50977249]]

        Y_star_std_expected = [[0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.]]

        np.testing.assert_array_almost_equal(Y_star_actual, Y_star_expected, decimal=2)
        np.testing.assert_allclose(Y_star_std_actual, Y_star_std_expected, atol=1e-3)

    def test_support_vector_regressor_valid(self):
        model = SurrogateModel(model_type='svr', X_names=self.input_names, Y_names=self.output_names,
                               X_range=self.ranges)

        model.fit(self.X, self.Y)
        model.save('test_model')

        model2 = SurrogateModel.load('test_model')
        pred = model2.predict(self.X_star, return_std=True)

        Y_star_actual = pred[0]
        Y_star_std_actual = pred[1]

        Y_star_expected = [[0.45188043, 0.60624406, 0.18685256],
                           [0.26052007, 0.4577048, 0.0313961],
                           [0.57807312, 0.15404722, 0.36128033],
                           [0.54474165, 0.42785959, 0.70173349],
                           [0.4371124, -0.03128216, 0.08514464],
                           [0.4387891, 0.86792197, 0.43871166],
                           [0.28427589, -0.04219891, 0.64624631],
                           [0.25383689, 0.05366502, 0.62013796],
                           [0.83442744, 0.63809326, 0.27550092],
                           [0.0880979, 0.66635484, 0.56484811]]

        np.testing.assert_array_almost_equal(Y_star_actual, Y_star_expected)
        self.assertIsNone(Y_star_std_actual)

    def test_polynomial_chaos_expansion_valid(self):
        model = SurrogateModel(model_type='pce', X_names=self.input_names, Y_names=self.output_names,
                               X_range=self.ranges)

        model.fit(self.X, self.Y)
        model.save('test_model')

        model2 = SurrogateModel.load('test_model')
        pred = model2.predict(self.X_star, return_std=True)

        Y_star_actual = pred[0]
        Y_star_std_actual = pred[1]

        Y_star_expected = [[0.4319164, 0.5633834, 0.1637688],
                           [0.24131087, 0.42753156, -0.00743406],
                           [0.55061594, 0.14019141, 0.34622029],
                           [0.52200876, 0.41781013, 0.67027579],
                           [0.42894579, -0.09285699, 0.04606886],
                           [0.42358707, 0.81266273, 0.41836294],
                           [0.26049254, -0.10449457, 0.62123501],
                           [0.23887531, 0.02511123, 0.59258566],
                           [0.80223692, 0.6068914, 0.24695545],
                           [0.05838493, 0.63603286, 0.54376015]]

        np.testing.assert_array_almost_equal(Y_star_actual, Y_star_expected)
        self.assertIsNone(Y_star_std_actual)

    def test_surrogate_model_invalid(self):
        model = SurrogateModel(model_type='gpr', X_names=self.input_names, Y_names=self.output_names,
                               X_range=self.ranges)

        X_alt_dim = np.random.rand(100, 2)
        Y_alt_dim = X_alt_dim ** 2

        X_alt_len = np.random.rand(90, 3)
        Y_alt_len = X_alt_len ** 2

        #  incorrect X labels shape
        def incorrect_X_labels_shape():
            model.X_names = self.input_names[:1]

        self.assertRaises(TypeError, incorrect_X_labels_shape)

        #  incorrect Y labels shape
        def incorrect_Y_labels_shape():
            model.Y_names = self.input_names[:1]

        self.assertRaises(TypeError, incorrect_Y_labels_shape)

        #  incorrect X dimension
        self.assertRaises(TypeError, model.fit, x=X_alt_dim, y=self.Y)

        #  incorrect Y dimension
        self.assertRaises(TypeError, model.fit, x=self.X, y=Y_alt_dim)

        #  incorrect Y length
        self.assertRaises(ValueError, model.fit, x=self.X, y=Y_alt_len)

        model.fit(self.X, self.Y)

        #  incorrect X dimension
        self.assertRaises(TypeError, model.predict, x=X_alt_dim)


if __name__ == '__main__':
    unittest.main()
