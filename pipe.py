from sklearn.pipeline import Pipeline, _fit_transform_one
import six

from sklearn.base import clone
from sklearn.externals import six
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_memory


# Override pipeline fit in order to authorize both X and y alteration in the fit process
# Modifies as well the scoring method (peculiar to SMOTE or other data augmentation alterations),
# As those steps needs to be bypassed when scoring the estimator (test data must not be changed).

class MyPipeline(Pipeline):

    def __init__(self, steps, memory=None):
        super(MyPipeline, self).__init__(steps=steps, memory=memory)

    def _fit(self, X, y=None, **fit_params):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            if transformer is None:
                pass
            else:
                if hasattr(memory, 'cachedir') and memory.cachedir is None:
                    # we do not clone when caching is disabled to preserve
                    # backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transfomer
                Xt, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, None, Xt, y,
                    **fit_params_steps[name])

                #########################################################
                if type(Xt) == tuple:
                    Xt, y = Xt
                #########################################################

                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)

        if self._final_estimator is None:
            #########################################################
            return Xt, y, {}
        return Xt, y, fit_params_steps[self.steps[-1][0]]
        #########################################################

    def fit(self, X, y=None, **fit_params):
        #########################################################
        Xt, y, fit_params = self._fit(X, y, **fit_params)
        #########################################################
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, y, **fit_params)
        return self

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, sample_weight=None):
        Xt = X
        #########################################################
        # The SMOTE has to be removed from pipeline when performing
        # the scoring (or evaluation).
        _steps = self.steps.copy()
        for (i, o) in enumerate(self.named_steps.keys()):
            if str(o) == 'smote':
                _steps.pop(i)
        #########################################################
        for name, transform in _steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, y, **score_params)
