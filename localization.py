# External imports
import numpy as np
from scipy.ndimage import shift

# Local imports
from config import APPS


class BayesFilter:
    def __init__(self, world):
        self.p_hit = 0.6
        self.p_miss = 0.2
        self.world = np.array(world)
        self.p = self.uniform_distribution(len(world))

    def control_update(self, offset=1):
        """
        Update the probability distribution based on movement using a Bayesian filter prediction step.

        Parameters:
        offset:
            - movement to the right: positive integer
            - movement to the left: negative integer
        """
        kernel = [0.1, 0.8, 0.1]
        self.p = self.exact_shift_and_rebalance(self.p, offset)
        return np.convolve(self.p, kernel, mode="same")

    def measurement_update(self, measurement):
        """
        Update the probability distribution based on a sensor measurement.

        Second step: Multiplication. This is the application of the Bayes Rule.
        P(Xi|Z) = (P(Z|Xi) * P(Xi)) / P(Z)
           - P(Xi|Z): Posterior
           - P(Z|Xi): Likelihood
           - P(Xi): Prior
           - P(Z): The normalization factor. P(Z) = SUM(P(Z|Xi) * P(Xi))
        Where:
           - Z: measurement
           - Xi: grid cell

        Parameters:
        - probs: Array of probabilities for each class.
        - measurement_idx: Index of the class where the sensor detected a measurement.
        - p_hit: Probability factor to adjust the probability of the detected class (correct measurement).
        - p_miss: Probability factor to adjust the probabilities of all other classes (incorrect measurement).

        Returns:
        - Updated probability array normalized to sum to 1.
        """
        assert measurement in self.world
        bool_selector = self.world == measurement
        self.p[bool_selector] *= self.p_hit
        self.p[~bool_selector] *= self.p_miss

        # Normalize to get the posterior probability distribution
        self.p /= self.p.sum()

    @staticmethod
    def uniform_distribution(n):
        """ """
        return np.ones(n) / n

    @staticmethod
    def exact_shift_and_rebalance(arr, movement):
        """ """
        assert np.count_nonzero(arr) > 0

        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=np.float32)

        arr = shift(arr, movement, mode="constant", cval=0)

        if movement < 0 and np.all(np.isclose(arr, 0)):
            arr[0] = 1
        elif movement > 0 and np.all(np.isclose(arr, 0)):
            arr[-1] = 1

        # Normalize because values have been discarded at the extremes
        arr /= arr.sum()
        return arr

    def get_prediction(self):
        return self.world[np.argmax(self.p)]


if __name__ == "__main__":
    world = ["Configuration", "Origin", "Search", "Apps"] + list(APPS.values())

    bayes_filter = BayesFilter(world)
    # Measurements. Each element of this array corresponds to one predicted class
    for movement, sensed_class in [
        (-1, "Apps"),
        (1, "Youtube"),
        (0, "Youtube"),
        (0, "Television"),
        (0, "Youtube"),
        (0, "Youtube"),
        (1, "Netflix"),
        (0, "Television"),
    ]:
        bayes_filter.control_update(movement)
        bayes_filter.measurement_update(sensed_class)
        index = np.argmax(bayes_filter.p)
        print(world[index])
