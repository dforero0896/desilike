"""Learned probability densities over a set of parameters.

A :class:`Density` is the non-Gaussian generalisation of
:class:`~desilike.samples.Covariance`.  A covariance reduces a chain to a mean and a
matrix; a density keeps the *shape* of the distribution -- curved degeneracies, skewness,
multi-modality -- and can be evaluated at an arbitrary point.  That is exactly what a
compressed likelihood needs: the ShapeFit chain is reduced to ``mean``/``covariance`` today
(see ``nb/compression_examples.ipynb``), which is a good approximation for BAO and a poor
one when ``df`` and ``dA`` trace out a banana.

Every density here is a *pure function* of a JAX array: ``log_prob`` is ``jit``/``grad``/
``vmap``-clean and float64, so it can sit inside a compiled desilike graph and be
differentiated through (which is what makes NUTS and the Optax profiler usable with a
non-Gaussian compressed likelihood).

Classes
-------
Density
    Abstract base: ``log_prob``, ``sample``, ``to_covariance``, serialization.

Functions
---------
chain_to_array
    ``(MCSamples, parameters)`` -> ``(X, weights, names)``.
prepare_training_samples
    Burn-in removal, thinning by integrated autocorrelation time, weight resampling.
fit_density
    Dispatcher: fit a density of a given ``kind`` to samples.
"""

import json
import logging

import numpy as np

from ..parameter import Variable, VariableCollection
from ..utils import register_type, write as _utils_write, read as _utils_read


def chain_to_array(samples, parameters):
    """Return ``(X, weights, names)`` from a chain and a list of parameter names.

    Parameters
    ----------
    samples : MCSamples or array
        A chain (any leading sample shape; it is ravelled) or a plain ``(nsamples, ndim)``
        array.  When an array is given, *parameters* is used only to label the columns.
    parameters : list of str
        Column names, in the order the density should use.  This order is the contract
        with the observable: ``ObservablesDensityLikelihood`` feeds it ``flattheory``,
        which is built from the observable's own ``parameters`` list.

    Returns
    -------
    X : numpy.ndarray, shape (nsamples, ndim)
    weights : numpy.ndarray, shape (nsamples,)
    names : list of str
    """
    names = [param if isinstance(param, str) else param.name for param in parameters]
    if isinstance(samples, np.ndarray):
        X = np.asarray(samples, dtype='f8')
        if X.ndim != 2:
            raise ValueError(f'expected a 2-D array of samples, got shape {X.shape}')
        if X.shape[1] != len(names):
            # Tolerate the transposed layout, but only when it is unambiguous: guessing here
            # silently fits the density to the wrong axes, and the likelihood then compares
            # qpar against a column of dA.
            if X.shape[0] == len(names):
                X = X.T
            else:
                raise ValueError(f'expected an array with {len(names)} columns for {names}, '
                                 f'got shape {X.shape}')
        return X, np.ones(X.shape[0]), names
    from ..samples.chain import MCSamples
    if not isinstance(samples, MCSamples):
        raise TypeError(f'expected MCSamples or an array, got {type(samples).__name__}')
    flat = samples.ravel()
    missing = [name for name in names if name not in flat.names()]
    if missing:
        raise ValueError(f'chain has no parameter(s) {missing}; it holds {flat.names()}')
    columns = [np.asarray(flat[name].value).reshape(flat.size, -1) for name in names]
    X = np.concatenate(columns, axis=1).astype('f8')
    if X.shape[1] != len(names):
        raise ValueError(f'parameters {names} must be scalar in the chain, got {X.shape[1]} columns')
    return X, np.asarray(flat.weight, dtype='f8').ravel(), names


def prepare_training_samples(samples, parameters, burnin=0., thin='auto', nsamples=None,
                             weighted='resample', rng=None, mpicomm=None):
    """Turn a chain into the unweighted ``(nsamples, ndim)`` array a density is fitted to.

    A chain is not a set of independent draws, and the two defects matter differently:
    correlation inflates how much a fit trusts the samples, and unequal weights (importance
    resampling, SMC, a reweighted chain) are simply not something a maximum-likelihood fit
    consumes.  Both are handled here, once, so every backend sees the same input.

    Parameters
    ----------
    samples : MCSamples or array
    parameters : list of str
    burnin : float, default=0.
        Fraction (in ``(0, 1)``) or number of leading steps to drop, as
        :meth:`~desilike.samples.chain.MCSamples.remove_burnin`.
    thin : int or 'auto' or None, default='auto'
        ``'auto'`` thins by the integrated autocorrelation time, so the retained samples
        are roughly independent.  An int thins by that factor; None keeps everything.
    nsamples : int, default=None
        Cap on the number of training samples (subsampled after thinning).  Fitting a flow
        to 10^6 correlated points is slower and no better than to 10^4 independent ones.
    weighted : str, default='resample'
        What to do with non-uniform weights: ``'resample'`` draws an equally weighted set
        with :func:`~desilike.samples.diagnostics.systematic_resample` (the only route a
        backend that maximises an unweighted likelihood can use), ``'keep'`` returns them
        for a backend that supports a weighted loss (the Gaussian mixture does), ``'ignore'``
        drops them.
    rng : int or numpy.random.Generator, default=None
    mpicomm : MPI communicator, optional
        When given, samples are gathered first, so every rank fits the same density.

    Returns
    -------
    X : numpy.ndarray, shape (nsamples, ndim)
    weights : numpy.ndarray or None
    names : list of str
    info : dict
        Bookkeeping: ``nsamples_raw``, ``nsamples``, ``ess``, ``iat``, ``kish_ess``.
    """
    from ..samples.diagnostics import integrated_autocorrelation_time, kish_ess, systematic_resample

    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(int(rng))
    if mpicomm is not None:
        samples = samples.gather(mpicomm=mpicomm)
    info = {}
    if isinstance(samples, np.ndarray):
        X, weights, names = chain_to_array(samples, parameters)
        info['nsamples_raw'] = X.shape[0]
    else:
        info['nsamples_raw'] = samples.size
        if burnin:
            samples = samples.remove_burnin(burnin)
        if thin == 'auto':
            try:
                # One value per parameter, so thin by the slowest: thinning by the average would
                # leave the worst-mixed parameter correlated.  `float()` on the returned array
                # raised, the except below swallowed it, and thinning was silently skipped --
                # which is worse than not offering the option, because the caller believes it
                # happened.
                iat = np.atleast_1d(np.asarray(integrated_autocorrelation_time(samples, params=parameters),
                                               dtype='f8'))
                iat = float(np.max(iat[np.isfinite(iat)])) if np.any(np.isfinite(iat)) else 1.
            except Exception as error:  # a short or badly mixed chain simply cannot be thinned
                logging.getLogger(__name__).warning(f'could not estimate the autocorrelation time ({error}); not thinning')
                iat = 1.
            thin = max(1, int(np.ceil(max(iat, 1.))))
            info['iat'] = iat
            info['ess'] = samples.size / thin
            logging.getLogger(__name__).info(
                f'auto-thinning by tau={thin} (slowest of {parameters}): {samples.size} samples, '
                f'ESS ~ {samples.size / thin:.0f}')
        elif thin is None:
            thin = 1
        samples = samples[::int(thin)]
        X, weights, names = chain_to_array(samples, parameters)
    info['nsamples'] = X.shape[0]
    uniform = np.allclose(weights, weights[0]) if weights.size else True
    if not uniform:
        info['kish_ess'] = kish_ess(np.log(np.clip(weights, 1e-300, None)))
    if nsamples is not None and X.shape[0] > nsamples:
        if uniform or weighted == 'ignore':
            index = rng.choice(X.shape[0], nsamples, replace=False)
        else:
            index = systematic_resample(weights, nsamples, rng)
            weights = np.ones(nsamples)
            uniform = True
        X, weights = X[np.sort(index)], weights[np.sort(index)]
        info['nsamples'] = X.shape[0]
    if weighted == 'resample' and not uniform:
        target = min(X.shape[0], int(info.get('kish_ess', X.shape[0])))
        index = systematic_resample(weights, max(target, 1), rng)
        X, weights, uniform = X[index], np.ones(index.size), True
        info['nsamples'] = X.shape[0]
    return X, (None if weighted != 'keep' or uniform else weights), names, info


@register_type
class Density:
    """A probability density over a named set of parameters, learned from samples.

    Attributes
    ----------
    names : list of str
        Parameter names, in the order the density's input vector is built.  This is the
        contract with the observable that feeds it.
    loc, scale : numpy.ndarray, shape (ndim,)
        The standardization the density was fitted in.  ``log_prob`` takes *physical*
        values and undoes this internally; a flow fitted to unstandardized compressed
        parameters (spreads of order 1e-2) simply does not train.
    limits : numpy.ndarray, shape (ndim, 2) or None
        The support box, from the first-stage priors when they are known.  Outside it the
        density returns ``-inf`` -- which is not a numerical detail: the chain cannot leave
        its own prior box, so any density fitted to it is unconstrained there, and letting
        it leak past the prior is exactly the error the Gaussian mean/covariance reduction
        makes.
    attrs : dict
        Fitting bookkeeping (``nsamples``, ``ess``, held-out ``nll``, the config needed to
        rebuild the architecture), serialized as JSON.
    """

    _name = 'Density'

    def __init__(self, names, loc=None, scale=None, limits=None, attrs=None):
        self.names = [name if isinstance(name, str) else name.name for name in names]
        self.ndim = len(self.names)
        self.loc = np.zeros(self.ndim) if loc is None else np.asarray(loc, dtype='f8')
        self.scale = np.ones(self.ndim) if scale is None else np.asarray(scale, dtype='f8')
        self.limits = None if limits is None else np.asarray(limits, dtype='f8').reshape(self.ndim, 2)
        self.attrs = dict(attrs or {})

    # ── the interface every backend implements ────────────────────────────────

    def _log_prob_standardized(self, x):
        """Log-density of *x*, already in standardized coordinates.  Override this.

        Convention for every backend: **fitted parameters are held in the standardized
        coordinates this method receives**, and :meth:`mean` / :meth:`covariance` return
        physical units.  Standardizing is not cosmetic -- a flow or a mixture fitted to
        compressed parameters, whose spreads are of order 1e-2 with a covariance whose
        condition number is large, will not train in physical units.
        """
        raise NotImplementedError

    def log_prob(self, x):
        """Log-density at *x*, a single point in physical units, shape ``(ndim,)``.

        The change of variables from the standardized coordinates contributes
        ``-sum(log(scale))``; a flow's own log-determinant is handled inside
        ``_log_prob_standardized``, which is where the two must not be confused -- dropping
        a log-det biases precisely the curved directions the density was fitted for.
        """
        import jax.numpy as jnp
        x = jnp.asarray(x)
        if x.shape != (self.ndim,):
            # Fail at the boundary, not three frames down inside a linear solve.  A likelihood
            # node handed a batch by mistake would otherwise broadcast into something shaped
            # like an answer, and the posterior would be quietly wrong.
            raise ValueError(f'{type(self).__name__}.log_prob takes a single point of shape '
                             f'({self.ndim},), got {x.shape}; use log_prob_batch for a batch')
        inside = True if self.limits is None else jnp.all((x >= self.limits[:, 0]) & (x <= self.limits[:, 1]))
        logpdf = self._log_prob_standardized((x - self.loc) / self.scale) - jnp.sum(jnp.log(self.scale))
        return jnp.where(inside, logpdf, -jnp.inf)

    def log_prob_batch(self, X):
        """``log_prob`` vmapped over the leading axis of *X*."""
        import jax
        return jax.vmap(self.log_prob)(X)

    def sample(self, key, shape):
        """Draw samples; used for posterior-predictive checks, not by the likelihood."""
        raise NotImplementedError(f'{type(self).__name__} does not implement sample()')

    def to_covariance(self, params=None):
        """Moment-matched :class:`~desilike.samples.Covariance`, for diagnostics and plots.

        Not what the likelihood uses -- the whole point is that this throws the shape away.
        Useful to hand to an observable that expects one, and to check a fitted density
        against the chain's own covariance.
        """
        from ..samples.covariance import Covariance
        params = [Variable(name) for name in self.names] if params is None else params
        return Covariance(np.asarray(self.covariance()), params=params)

    def covariance(self):
        """The density's covariance matrix, shape ``(ndim, ndim)``."""
        raise NotImplementedError

    def mean(self):
        """The density's mean, shape ``(ndim,)``."""
        raise NotImplementedError

    # ── diagnostics ──────────────────────────────────────────────────────────

    def priors(self):
        """The first-stage priors this density was fitted with, as ``{name: ParameterPrior}``.

        A parameter whose prior was improper or absent maps to ``None``; the whole result is
        ``None`` when the density was fitted from something that carried no priors at all.
        """
        from ..parameter import ParameterPrior
        raw = self.attrs.get('priors')
        if not raw:
            return None
        return {name: (None if state is None else ParameterPrior(**state))
                for name, state in json.loads(raw).items()}

    def coverage(self, samples, levels=(0.68, 0.95, 0.99), ncalibrate=20000, seed=0):
        """Fraction of *samples* falling inside the density's own nominal credible regions.

        The region threshold is calibrated **under the density**, from its own draws, rather
        than under the training data: ``tau_alpha`` solves ``P_q(log q >= tau) = alpha``, and
        the reported value is the fraction of *samples* with ``log q >= tau``.  That makes the
        check independent of any train/test bookkeeping -- which matters once ``fit_density``
        owns the split internally -- and it is the question a likelihood actually needs
        answered: do *my* credible regions hold *the data's* mass?

        It reproduces ``alpha`` when the density is the distribution the samples come from
        (checked to +-0.01 against the density's own fresh draws).  Note that this statistic and
        the older train/test-quantile version answer *different* questions: this one calibrates
        the model's own credible regions against data, that one checks whether log q distributes
        identically on the fit half and the held-out half (an overfitting detector).  For a
        well-specified fit they agree; when they disagree loudly -- on the ShapeFit chain's
        4-component fit, 1.000 against 0.541 at 68% -- that disagreement is itself the finding:
        the density is mis-specified (spikes plus a diffuse component), and its nominal regions
        are not the sets the data concentrate in.  Report both, and read the gap as a fit-quality
        alarm rather than as a coverage number.

        *samples* may be a chain (or list of chains) as readily as an array -- the diagnostic
        consumes what :func:`fit_density` consumes -- and is reduced over ``self.names``, so
        column order can never disagree with the density's.  A chain is used as given: its
        autocorrelation biases the *error bars* of the estimate, not its expectation.

        Calibrating and measuring on the same samples the density was fitted to is double
        dipping and reads optimistically: fit on one half of the chain and call this on the
        other.  Requires :meth:`sample`.
        """
        import jax
        import jax.numpy as jnp
        if hasattr(samples, 'names') and callable(getattr(samples, 'names')):
            samples, _, _ = chain_to_array(samples, self.names)      # an MCSamples
        elif isinstance(samples, (list, tuple)):
            from ..samples.chain import MCSamples
            samples, _, _ = chain_to_array(MCSamples.concatenate(*samples), self.names)
        else:
            samples = np.asarray(samples)
        draws = np.asarray(self.sample(jax.random.key(seed), (int(ncalibrate),)))
        logq_draws = np.asarray(self.log_prob_batch(jnp.asarray(draws)))
        logq_samples = np.asarray(self.log_prob_batch(jnp.asarray(samples)))
        return {str(level): float(np.mean(logq_samples >= np.quantile(logq_draws, 1. - level)))
                for level in levels}

    def heldout_nll(self, X, weights=None):
        """Mean negative log-likelihood on held-out *X*, in nats per point."""
        logpdf = np.asarray(self.log_prob_batch(X))
        if weights is None:
            return -float(np.mean(logpdf))
        weights = np.asarray(weights, dtype='f8')
        return -float(np.sum(weights * logpdf) / np.sum(weights))

    def gaussian_gap(self, X, weights=None):
        """How many nats/point the Gaussian mean/covariance reduction loses on *X*.

        The number to quote when justifying a non-Gaussian likelihood: ``0`` means the
        Gaussian was fine, and a few nats per point is a lot -- it is the log-likelihood
        difference accumulated at *every* point the second-stage sampler visits.
        """
        from .gmm import GaussianDensity
        baseline = GaussianDensity.fit(X, names=self.names, weights=weights)
        return self.heldout_nll(X, weights) - baseline.heldout_nll(X, weights)

    # ── serialization ────────────────────────────────────────────────────────

    def arrays(self):
        """The fitted numbers, as a ``{name: array}`` dict.  Override per backend."""
        return {}

    def _set_arrays(self, arrays):
        """Restore what :meth:`arrays` returned.  Override per backend."""
        raise NotImplementedError

    def _restore_attrs(self, attrs):
        """Hook, called with the restored *attrs* immediately **before** :meth:`_set_arrays`.

        A backend whose arrays are only interpretable given metadata -- the flow's
        architecture config, say -- reads it here, because ``_set_arrays`` runs from inside
        ``__setstate__`` and cannot wait for the subclass to finish restoring itself.
        """

    @classmethod
    def _cache_is_valid(cls, filename, config):
        """Is there a readable cache at *filename* whose recorded config matches *config*?

        ``build()`` re-runs every ``__init__`` and ``__post_init__``, so a fit that is not
        cached runs again on each build -- and a fit whose inputs changed must *not* be
        served from a stale file.  Both are handled by comparing the config the cache
        records against the one requested.
        """
        import os
        if filename is None or not os.path.exists(str(filename)):
            return False
        try:
            cached = _utils_read(filename)
        except Exception as error:
            logging.getLogger(__name__).warning(f'ignoring unreadable density cache {filename}: {error}')
            return False
        if not isinstance(cached, cls):
            return False
        try:
            return json.loads(cached.attrs.get('config', '{}')) == config
        except (ValueError, TypeError):
            return False

    def __getstate__(self, to_file=False):
        state = dict(self.arrays())
        state['loc'], state['scale'] = self.loc, self.scale
        if self.limits is not None:
            state['limits'] = self.limits
        if to_file:
            state['attrs'] = {'__class__': self._name, 'names': json.dumps(self.names),
                              **{key: str(value) for key, value in self.attrs.items()}}
        else:
            state['attrs'] = {**dict(self.attrs), 'names': self.names}
        return state

    def __setstate__(self, state):
        state = dict(state)
        attrs = state.pop('attrs', {})
        names = attrs.pop('names', None)
        self.names = json.loads(names) if isinstance(names, str) else list(names or [])
        self.ndim = len(self.names)
        self.loc = np.asarray(state.pop('loc'))
        self.scale = np.asarray(state.pop('scale'))
        self.limits = np.asarray(state['limits']) if 'limits' in state else None
        state.pop('limits', None)
        self.attrs = {key: value for key, value in attrs.items() if key != '__class__'}
        self._restore_attrs(self.attrs)
        self._set_arrays({key: np.asarray(value) for key, value in state.items()})

    def write(self, filename):
        """Write to an HDF5 (``.h5``) or text (``.txt``) file."""
        _utils_write(filename, self)

    @classmethod
    def read(cls, filename):
        """Read a density written by :meth:`write` (the subclass is recovered automatically).

        A backend registers itself with :func:`~desilike.utils.register_type` when its module
        is imported, and the flow backend is imported lazily so that reading a mixture never
        requires flowjax.  Importing the optional backends and retrying once keeps that
        property while still letting a flow cache be read by name -- without this, only the
        process that fitted the flow could read it back, which is precisely the cached path
        the design depends on.
        """
        try:
            return _utils_read(filename)
        except ValueError as error:
            if 'Unknown class' not in str(error):
                raise
            from . import flow  # noqa: F401  -- importing registers FlowDensity
            return _utils_read(filename)

    def validation_split(self, samples):
        """Recompute, exactly, the ``(train, val)`` split this density was fitted on.

        The split is a pure function of the samples and of ``seed``, ``thin`` and ``val_prop``
        -- all recorded in the fit config -- so it is rebuilt on demand rather than stored:
        the file stays a model artifact carrying no data, while the held-out set behind
        ``val_nll`` and the ``coverage_*`` attrs remains reproducible, which storing it was
        meant to achieve and does not achieve any better.

        Raises if the fit used ``seed=None``, where the split was never reproducible in the
        first place, and if the density carries no config (fitted by hand).
        """
        config = json.loads(self.attrs.get('config') or '{}')
        if not config:
            raise ValueError('this density carries no fit config; the split cannot be rebuilt')
        if config.get('seed') is None:
            raise ValueError('the fit used seed=None, so its split was never reproducible; '
                             'pass a seed to fit_density if you need this')
        thin = config.get('thin')
        thin = None if thin in (None, 'None') else ('auto' if thin == 'auto' else int(thin))
        rng = np.random.default_rng(int(config['seed']))
        X, weights, names, info = prepare_training_samples(
            samples, self.names, rng=rng, thin=thin,
            weighted='keep' if config.get('kind') == 'gmm' else 'resample')
        nval = max(1, int(float(config.get('val_prop', 0.1)) * X.shape[0]))
        permutation = rng.permutation(X.shape[0])
        return X[permutation[nval:]], X[permutation[:nval]]

    def __repr__(self):
        return f'{type(self).__name__}(names={self.names})'


#: Backends ``fit_density`` knows about, ``kind`` -> ``(module, class name)``.  Lazy so that
#: importing a density never imports flowjax (an optional extra, and Python >= 3.11 only).
#: The flow architectures are registered under their own names as well, so
#: ``fit_density(..., kind='bnaf')`` dispatches straight to the flow backend; the name is
#: forwarded to ``FlowDensity.fit`` as the architecture (see ``fit_density``).
_KINDS = {
    'gaussian': ('.gmm', 'GaussianDensity'),
    'gmm': ('.gmm', 'GaussianMixtureDensity'),
    'flow': ('.flow', 'FlowDensity'),
    'maf': ('.flow', 'FlowDensity'),
    'coupling': ('.flow', 'FlowDensity'),
    'bnaf': ('.flow', 'FlowDensity'),
}

#: Those ``kind`` values that name a flow *architecture* rather than just the flow backend,
#: and must therefore be forwarded to ``FlowDensity.fit(architecture=...)``.
_FLOW_ARCHITECTURES = ('maf', 'coupling', 'bnaf')


def fit_density(samples, parameters, kind='gmm', limits=None, priors=None, val_prop=0.1,
                select_from=None, filename=None, seed=None, verbose=True, thin='auto',
                diagnostics=True, **options):
    """Fit a density to a chain (or an array of samples) and return it.

    Parameters
    ----------
    samples : MCSamples, list of MCSamples, or array
        The chain to learn from -- typically the first-stage ShapeFit chain.
    parameters : list of str
        Which compressed parameters, and in which order.  Must match the observable's
        ``parameters``.
    kind : str, default='gmm'
        The backend: ``'gaussian'`` (mean/covariance -- the reduction this replaces, kept as
        a baseline and a regression test), ``'gmm'``, or a flow -- ``'flow'``/``'maf'``,
        ``'coupling'`` or ``'bnaf'``, the last three naming the flowjax architecture directly.
        Flow capacity and training knobs (``transformer``, ``knots``, ``interval``,
        ``flow_layers``, ``nn_width``, ``nn_depth``, ``max_epochs``, ...) are ordinary options
        here, and ``select_from`` can scan architectures as well as capacities.
    limits : array, shape (ndim, 2), default=None
        The support box.  When *priors* is given it is read off them instead.
    priors : VariableCollection, dict or Calculator, default=None
        The *first-stage* priors on *parameters*: the support box here, and the prior division
        the likelihood applies at evaluation time.  Normally there is nothing to pass -- they
        are read off the chain, whose Parameters carry them through an HDF5 round trip (see
        :func:`resolve_priors`) -- and they are recorded in the fitted density's attrs, so a
        likelihood rebuilt later from the density alone still knows them.  Pass them explicitly
        only when fitting from a bare array, or to override what the chain says.
    val_prop : float, default=0.1
        Fraction held out to report the out-of-sample negative log-likelihood.
    select_from : list of dict, default=None
        Candidate hyper-parameter sets (e.g. ``[{'ncomponents': 16}, {'ncomponents': 64}]``);
        each is fitted and the best held-out NLL wins.  Model selection on training data
        alone is what makes a mixture collapse onto narrow spikes.
    filename : str, default=None
        When given, the fitted density is cached there and re-read instead of re-fitted if
        the file exists and its recorded config matches.  ``build()`` re-runs every
        ``__init__`` and ``__post_init__``, so an uncached fit would run again on each build.
    thin : int, 'auto' or None, default='auto'
        Thinning applied to a chain before fitting; ``'auto'`` thins by the integrated
        autocorrelation time.  Part of the cache key: thinning changes the training data, so
        a cache written with one thinning must not be served to another.
    diagnostics : bool, default=True
        Compute the coverage diagnostics at fit time -- the model-calibrated
        ``coverage_*`` and the train/test-consistency ``coverage_emp_*`` attrs, plus
        ``coverage_gap_68`` -- on the held-out split.  Cheap for the mixture and Gaussian
        backends; a flow pays for one ``sample`` call and two batched ``log_prob`` passes.
        Turn off only when even that is too slow in a loop.  The numbers, not the samples,
        are what get stored.
    seed : int, default=None
    verbose : bool, default=True
    **options
        Forwarded to the backend (``ncomponents``, ``knots``, ``max_epochs``, ...).

    Returns
    -------
    Density
    """
    import importlib

    if isinstance(samples, (list, tuple)):
        from ..samples.chain import MCSamples
        samples = MCSamples.concatenate(*samples)
    kind = str(kind).lower()
    module_name, class_name = _KINDS[kind]
    # `__package__`, not `__name__`: the kinds are relative to `desilike.densities`, and
    # resolving them against this module would look for `desilike.densities.base.gmm`.
    cls = getattr(importlib.import_module(module_name, package=__package__), class_name)
    logger = logging.getLogger(__name__)

    # `kind='bnaf'` dispatches to the flow backend, but the backend's own architecture default
    # is 'maf' -- without this forwarding the registry entry would be a lie, and the cache
    # would record an architecture the fit never used.  An explicit `architecture=` wins.
    if kind in _FLOW_ARCHITECTURES:
        options = {'architecture': kind, **options}
        if select_from is not None:
            select_from = [{'architecture': kind, **candidate} for candidate in select_from]

    rng = np.random.default_rng(seed)
    priors = resolve_priors(samples, parameters, priors)
    if limits is None and priors is not None:
        limits = _limits_from_priors(priors, [p if isinstance(p, str) else p.name for p in parameters])
    # Labelling and concatenating the raw columns is cheap; thinning and resampling (which
    # estimate an autocorrelation time) are not, and are skipped entirely on a cache hit.
    raw_X, _, _ = chain_to_array(samples, parameters)
    config = {'kind': kind,
              'parameters': [p if isinstance(p, str) else p.name for p in parameters],
              'seed': None if seed is None else int(seed),
              'val_prop': float(val_prop),
              'thin': None if thin is None else str(thin),
              'select_from': [dict(sorted(c.items())) for c in (select_from or [options])],
              'options': dict(sorted({key: str(value) for key, value in options.items()}.items())),
              'limits': None if limits is None else np.asarray(limits).ravel().tolist(),
              'data': _fingerprint(raw_X)}
    if filename is not None and cls._cache_is_valid(filename, config):
        if verbose:
            logger.info(f'reading cached {kind} density from {filename}')
        return cls.read(filename)

    X, weights, names, info = prepare_training_samples(samples, parameters, rng=rng, thin=thin,
                                                       weighted='keep' if kind == 'gmm' else 'resample')
    if limits is not None:
        # A box that does not bracket the samples is not a small numerical annoyance: every
        # point would be masked to -inf, the held-out NLL of every candidate would be +inf, and
        # the model selection below would find no winner at all.  Say so here, naming the
        # parameters, rather than returning None from a comparison that never fired.
        limits = np.asarray(limits, dtype='f8').reshape(len(names), 2)
        outside = [f'{name}: samples span [{X[:, index].min():.6g}, {X[:, index].max():.6g}] '
                   f'but the box is [{limits[index, 0]:.6g}, {limits[index, 1]:.6g}]'
                   for index, name in enumerate(names)
                   if X[:, index].min() < limits[index, 0] or X[:, index].max() > limits[index, 1]]
        if outside:
            raise ValueError('the support box does not contain the samples; ' + '; '.join(outside))
    nval = max(1, int(val_prop * X.shape[0]))
    permutation = rng.permutation(X.shape[0])
    X_val, w_val = X[permutation[:nval]], (None if weights is None else weights[permutation[:nval]])
    X_tr, w_tr = X[permutation[nval:]], (None if weights is None else weights[permutation[nval:]])

    candidates = select_from if select_from else [options]
    best, best_nll = None, np.inf
    for candidate in candidates:
        fitted = cls.fit(X_tr, names=names, weights=w_tr, limits=limits,
                         loc=X_tr.mean(axis=0), scale=X_tr.std(axis=0), **candidate)
        nll = fitted.heldout_nll(X_val, w_val)
        if verbose:
            logger.info(f'{kind} {candidate}: held-out NLL = {nll:.4f} nats/point')
        if nll < best_nll:
            best, best_nll = fitted, nll
    from .gmm import GaussianDensity
    gaussian_nll = GaussianDensity.fit(X_tr, names=names, loc=X_tr.mean(axis=0),
                                       scale=X_tr.std(axis=0)).heldout_nll(X_val, w_val)
    if best is None:
        raise RuntimeError(f'no {kind} density could be fitted: every candidate scored a '
                           'non-finite held-out NLL.  Check that the samples are finite and '
                           'that the support box brackets them.')
    if diagnostics:
        # Coverage is computed here, at fit time, on the same held-out set `val_nll` refers to,
        # and only the numbers are kept: a density file is a model artifact and should not carry
        # data, while the split itself is rebuildable via `Density.validation_split`.  Both
        # statistics, because their disagreement is the mis-specification alarm.
        try:
            calibrated = best.coverage(X_val)
            logq_tr = np.asarray(best.log_prob_batch(X_tr))
            logq_val = np.asarray(best.log_prob_batch(X_val))
            empirical = {str(level): float(np.mean(logq_val >= np.quantile(logq_tr, 1. - level)))
                         for level in (0.68, 0.95, 0.99)}
            best.attrs.update({f'coverage_{int(round(float(level) * 100))}': str(value)
                               for level, value in calibrated.items()})
            best.attrs.update({f'coverage_emp_{int(round(float(level) * 100))}': str(value)
                               for level, value in empirical.items()})
            best.attrs['coverage_gap_68'] = str(abs(calibrated['0.68'] - empirical['0.68']))
        except NotImplementedError:
            logger.info(f'{kind} cannot sample; skipping the fit-time coverage diagnostics')
    best.attrs.update({**info, 'config': json.dumps(config), 'val_nll': str(best_nll),
                       'gaussian_val_nll': str(gaussian_nll), 'gain_over_gaussian': str(gaussian_nll - best_nll)})
    if priors is not None:
        # Recorded so that a likelihood rebuilt from this density alone -- no chain, no pipeline
        # -- still knows what to divide by and where the support box came from.
        best.attrs['priors'] = json.dumps(
            {name: (getattr(param, 'prior', None).__getstate__()
                    if getattr(param, 'prior', None) is not None else None)
             for name, param in zip(names, priors)})
    if verbose:
        logger.info(f'Gaussian mean/covariance would give {gaussian_nll:.4f} nats/point: '
                    f'the {kind} density gains {gaussian_nll - best_nll:.3f} nats/point on {info["nsamples"]} samples')
        if gaussian_nll - best_nll < 0.05:
            logger.warning('the Gaussian approximation is nearly as good here; a non-Gaussian '
                           'likelihood buys little and costs robustness -- check that the '
                           'parameters really are non-Gaussianly degenerate')
    if filename is not None:
        best.write(filename)
        if verbose:
            logger.info(f'wrote {filename}')
    return best


def _fingerprint(X):
    """A short, stable digest of a sample array, for cache invalidation.

    The cache key has to move whenever the fitted answer would: the samples themselves, the
    column order, the thinning, the backend and its hyper-parameters, and the support box.
    Keying on the backend name alone -- the obvious first attempt -- serves a 16-component
    mixture back to a caller who asked for 8.
    """
    import hashlib
    X = np.ascontiguousarray(np.asarray(X, dtype='f8'))
    digest = hashlib.sha256(X.tobytes()).hexdigest()[:16]
    return f'{X.shape[0]}x{X.shape[1]}:{digest}'


def resolve_priors(samples, parameters, priors=None):
    """The first-stage priors on *parameters*, from the most authoritative source available.

    Resolution order:

    1. an explicit *priors*: a :class:`~desilike.parameter.VariableCollection`, a dict, or a
       Calculator (the first-stage likelihood, theory or observable), resolved through
       :func:`~desilike.base.get_params`;
    2. otherwise the chain's own :class:`~desilike.parameter.Parameter` objects.  These survive
       an HDF5 round trip -- ``Parameter.__getstate__(to_file=True)`` serializes ``prior`` into
       the file's JSON metadata and ``__setstate__`` rebuilds it -- so a chain read from disk is
       self-describing, and this is the usual path;
    3. otherwise ``None``: a bare array of samples carries no priors.

    Returns a :class:`~desilike.parameter.VariableCollection` aligned with *parameters*, or None.
    """
    names = [param if isinstance(param, str) else param.name for param in parameters]
    if priors is None and samples is not None and not isinstance(samples, np.ndarray):
        flat = getattr(samples, 'ravel', lambda: samples)()
        try:
            priors = VariableCollection([flat[name] for name in names])
        except KeyError:
            priors = None
    if priors is not None and not isinstance(priors, (VariableCollection, dict)):
        from ..base import get_params     # lazy: densities must not import the graph at call time
        priors = get_params(priors)
    if isinstance(priors, dict):
        priors = VariableCollection([priors[name] for name in names])
    if priors is None:
        return None
    missing = [name for name in names if name not in priors.names()]
    if missing:
        raise ValueError(f'priors given for the wrong parameters: missing {missing}, '
                         f'have {priors.names()}')
    return VariableCollection([priors[name] for name in names])


def _limits_from_priors(priors, names):
    """Support box from the first-stage priors, or None when any is unbounded.

    Only a *bounded* prior defines a support: a uniform box or a truncated normal does, a plain
    normal does not (its limits are infinite), and truncating a density to a box the prior never
    declared would discard legitimate mass.  Hence None -- an unbounded density -- as soon as one
    parameter's prior is unbounded: a box covering some parameters and not others is not
    something the mask and the flow's squeeze currently represent.
    """
    if isinstance(priors, dict):
        priors = VariableCollection([priors[name] for name in names])
    limits = []
    for name in names:
        candidate = priors[name]
        # a Parameter carries its prior; a bare ParameterPrior (restored from a density's attrs,
        # say) is its own prior
        prior = getattr(candidate, 'prior', None)
        prior = prior if prior is not None else (candidate if hasattr(candidate, 'dist') else None)
        if prior is None:
            return None
        low, high = prior.limits
        if not (np.isfinite(low) and np.isfinite(high)):
            return None
        limits.append((low, high))
    return np.asarray(limits, dtype='f8')
