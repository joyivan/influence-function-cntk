from abc import abstractmethod, abstractproperty
from sklearn.metrics import accuracy_score


class Evaluator(object):
    """Base class for evaluation functions."""

    @abstractproperty
    def name(self):
        """
        The name of evaluation metric.
        :return str.
        :return:
        """
        pass

    @abstractproperty
    def worst_score(self):
        """
        The worst performance score.
        :return float.
        """
        pass

    @abstractproperty
    def mode(self):
        """
        The mode for performance score, either 'max' or 'min'.
        e.g. 'max' for accuracy, AUC, precision and recall,
              and 'min' for error rate, FNR and FPR.
        :return: str.
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        Performance metric for a given prediction.
        This should be implemented.
        :param y_true: np.ndarray, shape: (N, num_classes).
        :param y_pred: np.ndarray, shape: (N, num_classes).
        :return float.
        """
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        """
        Function to return whether current performance score is better than current best.
        This should be implemented.
        :param curr: float, current performance to be evaluated.
        :param best: float, current best performance.
        :return bool.
        """
        pass


class ErrorRateEvaluator(Evaluator):
    """Evaluator with error rate metric."""

    @property
    def name(self):
        """The name of evaluation metric."""
        return 'error rate'

    @property
    def worst_score(self):
        """The worst performance score."""
        return 1.0

    @property
    def mode(self):
        """The mode for performance score."""
        return 'min'

    def score(self, y_true, y_pred):
        """Compute accuracy for a given prediction."""
        return 1. - accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    def is_better(self, curr, best, **kwargs):
        """
        Return whether current performance score is better than current best,
        with consideration of the relative threshold to the given performance score.
        :param kwargs: dict, extra arguments.
            - score_threshold: float, relative threshold for measuring the new optimum,
                               to only focus on significant changes.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 - score_threshold
        return curr < best * relative_eps
