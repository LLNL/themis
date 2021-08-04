from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from uqp.sampling import sampler
from scipy import optimize
from copy import deepcopy as cp

try:
    from sampadapt.Surrogates import DJINN
except:
    print("DJINN is not availible")
    DJINN = None


class AdaptiveSampler(sampler.ContinuousSampler):

    name = "Adaptive"

    @staticmethod
    def sample_points(**kwargs):
        """Returns a set of new sample points based on some set of states given"""
        pass


class ScoredSampler(AdaptiveSampler):
    """A Base Sampler for adaptive sampling based on score"""

    name = "Scored"

    @classmethod
    def sample_points(cls, **kwargs):
        """
        Creates a set of sample points from a set of candidate points based on an implemented score function.

        Classes that inherit from ScoredSampler must implement _get_score. This function is evaluated at all of
        the candidate points. The points with the top scores are returned. Must specify either nCand and box, or
        npCandPnts. Generated candidate points are created using LatinHyperCubeSampler.
        """

        if 'cand_points' in kwargs and kwargs['cand_points'] is not None:  # Use provided candidate points
            np_candidate_points = kwargs['cand_points']

        # Generate candidate points using LHS
        elif 'num_cand_points' in kwargs and kwargs['num_cand_points'] is not None:
            i_num_candidates = int(kwargs['num_cand_points'])
            ls_box = kwargs['box']
            np_candidate_points = sampler.LatinHyperCubeSampler.sample_points(num_points=i_num_candidates, box=ls_box)
            kwargs['cand_points'] = np_candidate_points

        else:
            raise TypeError("Neither 'cand_points' nor 'num_cand_points' was given")

        i_num_points = int(kwargs['num_points'])

        ls_scores = cls._get_score(**kwargs)  # Inheriting samplers must implement _get_score

        # Pick the points with the top scores (Sort the list and take the last N)
        indices = np.argsort(ls_scores)[-i_num_points:]  # negative index loops around to the end
        return np_candidate_points[indices]

    @staticmethod
    def _get_score(**kwargs):
        raise NotImplementedError


class WeightedSampler(AdaptiveSampler):
    """A Base Sampler for adaptive sampling based on weights"""

    name = "Weighted"

    @classmethod
    def sample_points(cls, **kwargs):
        """
        Creates a set of sample points from a set of candidate points based on an implemented weight function.

        Classes that inherit from WeightedSampler must implement _get_weights. This function is evaluated at all of
        the candidate points. Points are drawn randomly without replacement. A point's probability of being drawn is
        proportional to its weight. Must specify either nCand and box, or npCandPnts. Generated candidate points are
        created using LatinHyperCubeSampler.
        """

        if 'cand_points' in kwargs and kwargs['cand_points'] is not None:
            np_candidate_points = kwargs['cand_points']
        elif 'num_cand_points' in kwargs and kwargs['num_cand_points'] is not None:
            i_num_candidates = int(kwargs['num_cand_points'])
            ls_box = kwargs['box']
            np_candidate_points = sampler.LatinHyperCubeSampler.sample_points(num_points=i_num_candidates, box=ls_box)
            kwargs['cand_points'] = np_candidate_points
        else:
            raise TypeError("Neither 'cand_points' nor 'num_cand_points' was given")

        i_num_points = int(kwargs['num_points'])

        weights = cls._get_weights(**kwargs)  # Inheriting samplers must implement _get_weights()

        weights_sum = weights.sum()

        # Normalize weights to probability, then choose without replacement.
        # The probability of choosing a point is proportional to its weight
        indices = np.random.choice(len(weights), i_num_points, replace=False, p=weights/weights_sum)

        return np_candidate_points[indices]

    @staticmethod
    def _get_weights(**kwargs):
        raise NotImplementedError


class ActiveLearningSampler(ScoredSampler):
    """A Sampler for adaptive sampling based on standard deviation"""

    name = "Active Learning"

    @staticmethod
    def sample_points(num_points, model, num_cand_points=None, box=None, cand_points=None, **kwargs):
        """
        Creates a set of sample points based on the standard deviation of the surrogate model.

        Returns the candidate points that has the highest standard deviation in the surrogate model. Requires a
        surrogate model that can give standard deviation point estimate.

        Args:
            - num_points (int): The number of sample points to return (required)
            - num_cand_points (int): The number of candidate points to generate (optional)
            - box ([[float]]): The bounding box of the candidate points (optional)
            - cand_points ([[float]]): The set of candidate points
            - model (Surrogate Model): The trained surrogate model

        Returns (numpy array):
            - A two dimensional numpy array of sample points
        """
        return super(ActiveLearningSampler, ActiveLearningSampler).sample_points(num_points=num_points,
                                                                                 model=model,
                                                                                 num_cand_points=num_cand_points,
                                                                                 box=box,
                                                                                 cand_points=cand_points)

    @staticmethod
    def _get_score(**kwargs):
        """
        Score for each point is the standard deviation at that point

        Args:
            - kwargs:
                * model (Surrogate Model): The trained surrogate model
                * npCandPnts ([[float]]): The set of candidate points

        Returns ([float]):
            - Array of scores
        """

        model = kwargs['model']
        np_candidate_points = kwargs['cand_points']

        # Model must be able to give a standard deviation
        _, np_candidate_sigma = model.predict(np_candidate_points, return_std=True)
        return np_candidate_sigma.flatten()


class DeltaSampler(ScoredSampler):
    """A Sampler for adaptive sampling based on nearest neighbor delta"""

    name = "Delta"

    @staticmethod
    def sample_points(num_points, model, X, Y, num_cand_points=None, box=None, cand_points=None, **kwargs):
        """
        Creates a set of sample points based on the delta of nearest neighbor outputs.

        Returns the candidate points that has the largest difference between its output and the output of its nearest
        neighbor.

        Args:
            - num_points (int): The number of sample points to return (required)
            - num_cand_points (int): The number of candidate points to generate (optional)
            - box ([[float]]): The bounding box of the candidate points (optional)
            - cand_points ([[float]]): The set of candidate points
            - model (Surrogate Model): The trained surrogate model
            - X (numpy array): The input points for training 'model'
            - Y (numpy array): The output points for training 'model'

        Returns (numpy array):
            - A two dimensional numpy array of sample points
        """
        return super(DeltaSampler, DeltaSampler).sample_points(num_points=num_points,
                                                               model=model,
                                                               num_cand_points=num_cand_points,
                                                               box=box,
                                                               cand_points=cand_points,
                                                               X=X,
                                                               Y=Y)

    @staticmethod
    def _get_score(**kwargs):
        """
        Score for each point is the delta relative to the nearest neighbor.

        For each point returns the difference between its output and the output of its nearest neighbor.

        Args:
            - kwargs:
                * model (Surrogate Model): The trained surrogate model
                * cand_points ([[float]]): The set of candidate points
                * X (numpy array): The input points for training 'model'
                * Y (numpy array): The output points for training 'model'

        Returns ([float]):
            - Array of scores
        """

        model = kwargs['model']
        np_x = kwargs['X']
        np_y = kwargs['Y']
        np_candidate_points = kwargs['cand_points']

        np_candidate_predicted, _ = model.predict(np_candidate_points, return_std=False)

        # For each candidate point find the closest point in npX
        np_closest_indices = np.array([DeltaSampler._get_closest_index(np_candidate_points[i], np_x)
                                       for i in range(np_candidate_points.shape[0])])

        # Get the npY value from the corresponding npX value closest to each candidate points
        np_closest_values = np_y[np_closest_indices]
        return abs(np_closest_values-np_candidate_predicted).flatten()

    @staticmethod
    def _get_closest_index(reference_point, X):
        """
        Returns the index of the point closest to the reference point

        Args:
            - reference_point ([float]): The point from which to determine distance
            - X ([[float]]): The set of points on which to determine distance

        Returns (int):
            - The index of the point in npX closest to the reference point
        """
        np_distances = np.array([np.linalg.norm(reference_point - X[i]) for i in range(X.shape[0])])
        return np_distances.argmin()


class ExpectedImprovementSampler(ScoredSampler):
    """A Sampler for adaptive sampling based on a mix of nearest neighbor delta and standard deviation"""

    name = "Expected Improvement"

    @staticmethod
    def sample_points(num_points, model, X, Y, num_cand_points=None, box=None, cand_points=None, **kwargs):
        """
        Creates a set of sample points based on the ALM score and Delta score

        Returns the candidate points that maximize the L2 norm of the ALM score and the Delta score

        Args:
            - num_points (int): The number of sample points to return (required)
            - num_cand_points (int): The number of candidate points to generate (optional)
            - box ([[float]]): The bounding box of the candidate points (optional)
            - cand_points ([[float]]): The set of candidate points
            - model (Surrogate Model): The trained surrogate model
            - X (numpy array): The input points for training 'model'
            - Y (numpy array): The output points for training 'model'

        Returns (numpy array):
            - A two dimensional numpy array of sample points
        """
        return super(ExpectedImprovementSampler,
                     ExpectedImprovementSampler).sample_points(num_points=num_points,
                                                               model=model,
                                                               num_cand_points=num_cand_points,
                                                               box=box,
                                                               cand_points=cand_points,
                                                               X=X,
                                                               Y=Y)

    @staticmethod
    def _get_score(**kwargs):
        """
        Score for each point is the L2 norm of the ALM score and the Delta score.

        .. math::
            \\sqrt{ASM^2 + Delta^2}

        Args:
            - kwargs:
                * model (Surrogate Model): The trained surrogate model
                * npCandPnts ([[float]]): The set of candidate points
                * npX (numpy array): The input points for training 'model'
                * npY (numpy array): The output points for training 'model'

        Returns ([float]):
            - Array of scores
        """
        np_delta_score = DeltaSampler._get_score(**kwargs)
        np_active_score = ActiveLearningSampler._get_score(**kwargs)
        return np.sqrt(np.power(np_delta_score, 2.0) + np.power(np_active_score, 2.0))


class LearningExpectedImprovementSampler(ScoredSampler):
    """A Sampler for adaptive sampling based on learning a mix of nearest neighbor delta and standard deviation"""

    name = "Learning Expected Improvement"

    @staticmethod
    def sample_points(num_points, model, X, Y, num_cand_points=None, box=None, cand_points=None, **kwargs):
        """
        Creates a set of sample points based on learning the best relationship using the ALM score and Delta score

        Learns optimal weights (alpha, beta, rho) for Expected Improvement from training data.
        Scores are based on these learned weights as well as the ALM and Delta scores.

        .. math::
            \\alpha + \\beta(\\rho ALM^2 + (1 - \\rho)Delta^2)

        Args:
            - num_points (int): The number of sample points to return (required)
            - num_cand_points (int): The number of candidate points to generate (optional)
            - box ([[float]]): The bounding box of the candidate points (optional)
            - cand_points ([[float]]): The set of candidate points
            - model (Surrogate Model): The trained surrogate model
            - X (numpy array): The input points for training 'model'
            - Y (numpy array): The output points for training 'model'

        Returns (numpy array):
            - A two dimensional numpy array of sample points
        """
        return super(LearningExpectedImprovementSampler,
                     LearningExpectedImprovementSampler).sample_points(num_points=num_points,
                                                                       model=model,
                                                                       num_cand_points=num_cand_points,
                                                                       box=box,
                                                                       cand_points=cand_points,
                                                                       X=X,
                                                                       Y=Y)

    @staticmethod
    def _get_score(**kwargs):
        """
        Score for each point is the optimal combination of ALM and Delta scores learned from the training points.

        Optimizes alpha, beta, and rho on ALM and Delta scores through leave-one-out bootstrapping on the training
        points.

        Args:
            - kwargs:
                * cand_points ([[float]]): The set of candidate points
                * model (Surrogate Model): The trained surrogate model
                * X (numpy array): The input points for training 'model'
                * Y (numpy array): The output points for training 'model'

        Returns ([float]):
            - Array of scores
        """

        model = kwargs['model']
        np_x = kwargs['X']
        np_y = kwargs['Y']
        np_candidate_points = kwargs['cand_points']

        np_intermediate_ALM, np_intermediate_delta, f_residual = \
            LearningExpectedImprovementSampler._calculate_intermediate_scores(cp(model), np_x, np_y)

        def score(_alpha, _beta, _rho, _alm, _delta):
            return _alpha + _beta * (_rho * np.power(_alm, 2.0) + (1 - _rho) * np.power(_delta, 2.0))

        def error(ls_values):
            _alpha, _beta, _rho = ls_values
            return sum(np.power(f_residual - score(_alpha, _beta, _rho,
                                                   np_intermediate_ALM, np_intermediate_delta),
                                2.0))

        results = [None]*3
        for i, startWeight in enumerate([0.01, 0.5, 0.99]):
            results[i] = optimize.minimize(error,
                                     x0=[f_residual, 1.0, startWeight],
                                     method='L-BFGS-B',
                                     bounds=((None, None), (0.0, None), (0.0, 1.0)))

        ALM_score = ActiveLearningSampler._get_score(model=model, cand_points=np_candidate_points)
        delta_score = DeltaSampler._get_score(model=model, cand_points=np_candidate_points, X=np_x, Y=np_y)

        min_index = np.argmin([result['fun'] for result in results])

        alpha, beta, rho = results[min_index].x
        return score(alpha, beta, rho, ALM_score, delta_score)

    @staticmethod
    def _calculate_intermediate_scores(model, X, Y):
        """
        Calculates the intermediate scores for each point.

        Fits temporary surrogate models to training points, while leaving each training point out once. The ALM and
        Delta scores are calculated on the left out point. The residual is the difference between the known training
        value and the value predicted when that value is left out.

        Args:
            - model (Surrogate Model): The re-fittable surrogate model
            - X (numpy array): The input points for training 'model'
            - Y (numpy array): The output points for training 'model'

        Returns (3-tuple of numpy arrays):
            - Intermediate ALM, Delta, and Residual scores for each point
        """

        np_training_ALM = np.array([0.] * len(Y))
        np_training_delta = np.array([0.] * len(Y))
        np_residuals = np.array([0.] * len(Y))

        for index in range(len(Y)):

            # Construct model leaving out npX[index] and npY[index]
            left_out_point = X[index].reshape(1, -1)
            np_alt_x = np.delete(X, [index], 0)

            np_alt_y = np.delete(Y, [index], 0)

            if type(model) is DJINN:
                n_trees = model.regressor.__n_trees
                max_tree_depth = model.regressor.__tree_max_depth
                dropout_keep_prob= model.regressor.__dropout_keep_prob
                model = DJINN(n_trees=n_trees,
                              max_tree_depth=max_tree_depth,
                              dropout_keep_prob=dropout_keep_prob,
                              name='DJINN')

            model.fit(np_alt_x, np_alt_y)

            # Get scores from new model on left out point
            np_training_ALM[index] = ActiveLearningSampler._get_score(model=model, cand_points=left_out_point)[0]
            np_training_delta[index] = DeltaSampler._get_score(model=model, cand_points=left_out_point,
                                                               X=np_alt_x, Y=np_alt_y)[0]
            y_pred, _ = model.predict(left_out_point)
            np_residuals[index] = np.absolute(Y[index] - y_pred)

        np_mean_squared_error = np.power(np_residuals, 2.0)

        np_weight = np.mean(1./np_mean_squared_error)
        f_residual = np.sqrt(1./np_weight)

        return np_training_ALM, np_training_delta, f_residual
