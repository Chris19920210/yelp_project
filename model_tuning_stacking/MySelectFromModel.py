import numpy as np
from sklearn.feature_selection.base import SelectorMixin
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.externals import six
from sklearn.utils import safe_mask, check_array, deprecated
from sklearn.utils.validation import check_is_fitted
from numpy import median

"""
The MySelectFromModel is based on Sklearn's SelectFormModel, the modification
is trying to provide the data scientist a more convenient way to tune the
hyperparamaters. Namely, instead of setting the threshold for 'importance',
here we can directly tune the dimension of reduction by using methods such as
random forest. the default is to shrink the data set to half of its original
dimension based on variable importance.
"""

def _calculate_threshold(estimator, scores, n_components):
    lists = sorted(scores, reverse = True)
    if n_components == None:
        threshold = median(lists)
    else:
        threshold = lists[n_components-1]
    return threshold


def _get_feature_importances(estimator):
    """Retrieve or aggregate feature importances from estimator"""
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_

    elif hasattr(estimator, "coef_"):
        if estimator.coef_.ndim == 1:
            importances = np.abs(estimator.coef_)

        else:
            importances = np.sum(np.abs(estimator.coef_), axis=0)

    else:
        raise ValueError(
            "The underlying estimator %s has no `coef_` or "
            "`feature_importances_` attribute. Either pass a fitted estimator"
            " to SelectFromModel or call fit before calling transform."
            % estimator.__class__.__name__)

    return importances



class MySelectFromModel(SelectorMixin, BaseEstimator):
    def __init__(self, estimator, prefit=False, n_components = None):
        self.estimator = estimator
        self.n_components = n_components
        self.prefit = prefit
        self.feature_importance = None

    def _get_support_mask(self):

        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, 'estimator_'):
            estimator = self.estimator_
        else:
            raise ValueError('Either fit the model before transform or set "prefit=True"'' while passing the fitted estimator to the constructor.')
        scores = _get_feature_importances(estimator)
        self.feature_importance = scores
        self.threshold_ = _calculate_threshold(estimator, scores, self.n_components)

        return scores >= self.threshold_

    def fit(self, X, y=None, **fit_params):
        if not hasattr(self, "estimator_"):
            self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self

    def partial_fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer only once.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).
        **fit_params : Other estimator specific parameters
        Returns
        -------
        self : object
            Returns self.
        """
        if not hasattr(self, "estimator_"):
            self.estimator_ = clone(self.estimator)
        self.estimator_.partial_fit(X, y, **fit_params)
        return self

