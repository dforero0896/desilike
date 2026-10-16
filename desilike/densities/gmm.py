"""Gaussian and Gaussian-mixture densities -- pure ``jax.numpy``, no extra dependency.

The mixture is the backend to reach for first at the dimensionality compressed variables
live at (4 to 6).  On a 5-D toy with a curved ``dA``--``df``-style degeneracy and an
exactly known density, fitted to 1.2e4 samples, the held-out negative log-likelihood per
point was:

===============  =========  ==========================
model            NLL        excess over the true value
===============  =========  ==========================
truth            -13.353    --
Gaussian          -11.818   +1.534
GMM (16 comps)    -13.286   +0.067
GMM (64 comps)    -13.200   +0.153  (starts to overfit)
MAF (8 knots)     -12.995   +0.358
===============  =========  ==========================

So a 16-component mixture recovered ~96% of what the mean/covariance reduction throws
away, and beat a spline masked-autoregressive flow at that sample size.  The mixture also
has closed-form Gaussian tails (a sampler that wanders off gets a large-but-finite penalty
rather than a cliff), fits weighted samples natively, is deterministic, and serializes as
three arrays.  The flow earns its keep when the dimension grows or the manifold is genuinely
complicated; see :mod:`desilike.densities.flow`.
"""

import logging

import numpy as np

from ..utils import register_type
from .base import Density


@register_type
class GaussianDensity(Density):
    """Mean + covariance -- the reduction the compressed likelihoods use today.

    Here as the baseline every other backend is measured against, and as a regression test:
    on a Gaussian chain, :class:`~desilike.likelihoods.density.ObservablesDensityLikelihood`
    with this density must reproduce :class:`~desilike.likelihoods.ObservablesGaussianLikelihood`.

    Parameters
    ----------
    mean : array, shape (ndim,)
    covariance : array, shape (ndim, ndim)
    names : list of str
    """

    _name = 'GaussianDensity'

    def __init__(self, mean, covariance, names, **kwargs):
        super().__init__(names=names, **kwargs)
        # Private: `Density.mean()` and `Density.covariance()` are methods, and a subclass
        # that shadows either with an array breaks `to_covariance()` and `gaussian_gap()`.
        # The caller passes physical units; what is stored is standardized, per the
        # `Density._log_prob_standardized` convention, and `mean()`/`covariance()` map back.
        # Converting once here rather than per call keeps `loc`/`scale` out of the hot path.
        mean = np.asarray(mean, dtype='f8')
        covariance = np.atleast_2d(np.asarray(covariance, dtype='f8'))
        self._mean = (mean - self.loc) / self.scale
        self._covariance = covariance / self.scale[:, None] / self.scale[None, :]

    @classmethod
    def fit(cls, X, names, weights=None, **kwargs):
        X = np.asarray(X, dtype='f8')
        if weights is None:
            mean, cov = X.mean(axis=0), np.cov(X.T)
        else:
            weights = np.asarray(weights, dtype='f8')
            total = weights.sum()
            mean = (weights[:, None] * X).sum(axis=0) / total
            residual = X - mean
            cov = ((weights[:, None] * residual).T @ residual) * total / (total ** 2 - (weights ** 2).sum())
        return cls(mean=mean, covariance=np.atleast_2d(cov), names=names, **kwargs)

    def _log_prob_standardized(self, x):
        import jax
        import jax.numpy as jnp
        covariance = jnp.asarray(self._covariance)
        residual = jnp.asarray(x) - jnp.asarray(self._mean)
        solve = jax.scipy.linalg.solve(covariance, residual, assume_a='pos')
        _, logdet = jnp.linalg.slogdet(covariance)
        return -0.5 * jnp.dot(residual, solve) - 0.5 * logdet - 0.5 * self.ndim * jnp.log(2 * jnp.pi)

    def sample(self, key, shape):
        import jax
        import jax.numpy as jnp
        cholesky = jnp.linalg.cholesky(jnp.asarray(self._covariance))
        draws = jax.random.normal(key, tuple(shape) + (self.ndim,))
        return jnp.asarray(self.loc) + jnp.asarray(self.scale) * (jnp.asarray(self._mean) + draws @ cholesky.T)

    def mean(self):
        return self.loc + self.scale * self._mean

    def covariance(self):
        return self.scale[:, None] * self._covariance * self.scale[None, :]

    def arrays(self):
        return {'mean': self._mean, 'covariance': self._covariance}

    def _set_arrays(self, arrays):
        self._mean = arrays['mean']
        self._covariance = arrays['covariance']



@register_type
class GaussianMixtureDensity(Density):
    r"""A Gaussian mixture, fitted by (optionally weighted) EM.

    .. math:: \log q(x) = \log \sum_{k} w_k \, \mathcal{N}(x \,|\, \mu_k, \Sigma_k)

    Parameters
    ----------
    weights : array, shape (ncomponents,)
    means : array, shape (ncomponents, ndim)
    covariances : array, shape (ncomponents, ndim, ndim)
    names : list of str
    floor_weight : float, default=0.
        Weight of an extra broad Gaussian component covering the whole fit range, added as
        ``logaddexp(log q, log floor_weight + log N_broad)``.  A hard zero-density region is
        a wall a sampler cannot cross, and a gradient-based kernel gets a NaN there (desilike
        sets ``EQX_ON_ERROR=nan``, so the NaN propagates instead of raising).  Something of
        order 1e-4 to 1e-6 costs nothing in the bulk and keeps the posterior connected.  The
        chain under-samples its own tails, so *some* floor is honest rather than optimistic.

    Other Parameters
    ----------------
    covariance_floor : float, default=1e-6
        Added to each component covariance as a fraction of the global variance, per
        dimension.  Without it EM drives components onto narrow spikes: the held-out NLL
        keeps improving while the density becomes a sum of deltas, which is the failure mode
        that makes a mixture likelihood unusable for sampling.
    """

    _name = 'GaussianMixtureDensity'

    def __init__(self, weights, means, covariances, names, floor_weight=0., **kwargs):
        super().__init__(names=names, **kwargs)
        self.weights = np.asarray(weights, dtype='f8')
        self.means = np.atleast_2d(np.asarray(means, dtype='f8'))
        self.covariances = np.asarray(covariances, dtype='f8').reshape(len(self.weights), self.ndim, self.ndim)
        self.ncomponents = len(self.weights)
        self.floor_weight = float(floor_weight)

    # ── fitting ──────────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, X, names, weights=None, ncomponents=16, niter=300, tol=1e-6, covariance_floor=1e-6,
            floor_weight=0., seed=0, loc=None, scale=None, limits=None, **kwargs):
        """Fit by EM.  *X* is expected in physical units; it is standardized internally."""
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom

        logger = logging.getLogger(__name__)
        X = np.asarray(X, dtype='f8')
        loc = X.mean(axis=0) if loc is None else np.asarray(loc, dtype='f8')
        scale = X.std(axis=0) if scale is None else np.asarray(scale, dtype='f8')
        scale = np.where(scale > 0, scale, 1.)
        Z = jnp.asarray((X - loc) / scale)
        nsamples, ndim = Z.shape
        ncomponents = int(min(ncomponents, max(2, nsamples // 10)))
        responsibility_weights = jnp.ones(nsamples) if weights is None else jnp.asarray(weights, dtype='f8')
        if not jnp.all(jnp.isfinite(responsibility_weights)) or jnp.any(responsibility_weights <= 0):
            raise ValueError('sample weights must be positive and finite')

        key = jrandom.key(seed)
        chosen = jrandom.choice(key, nsamples, (ncomponents,), replace=False)
        means = Z[chosen]
        global_covariance = jnp.cov(Z.T) if ndim > 1 else jnp.ones((1, 1)) * Z.var()
        covariances = jnp.repeat(jnp.atleast_2d(global_covariance)[None], ncomponents, axis=0)
        mixing = jnp.full(ncomponents, 1. / ncomponents)
        floor = covariance_floor * jnp.diag(jnp.ones(ndim))

        def expectation_maximization_step(means, covariances, mixing):
            covariances = covariances + floor
            precisions = jnp.linalg.inv(covariances)
            _, logdets = jnp.linalg.slogdet(covariances)
            offset = Z[:, None, :] - means[None]
            mahalanobis = jnp.einsum('nkd,kde,nke->nk', offset, precisions, offset)
            logpdf = -0.5 * mahalanobis - 0.5 * logdets[None] - 0.5 * ndim * jnp.log(2 * jnp.pi) + jnp.log(mixing)[None]
            logpdf = logpdf + jnp.log(responsibility_weights)[:, None]
            maximum = logpdf.max(axis=1, keepdims=True)
            responsibility = jnp.exp(logpdf - maximum)
            responsibility = responsibility / responsibility.sum(axis=1, keepdims=True)
            counts = responsibility.sum(axis=0) + 1e-10
            means = jnp.einsum('nk,nd->kd', responsibility, Z) / counts[:, None]
            offset = Z[:, None, :] - means[None]
            covariances = jnp.einsum('nk,nkd,nke->kde', responsibility, offset, offset) / counts[:, None, None]
            mixing = counts / counts.sum()
            loglike = float(jnp.sum(maximum.ravel() + jnp.log(jnp.exp(logpdf - maximum).sum(axis=1))))
            return means, covariances, mixing, loglike

        previous, iteration = -np.inf, 0
        # The likelihood is only read off the device every `check_every` iterations: each read
        # is a host sync, and checking every step made the fit an order of magnitude slower
        # than the EM it was supervising.
        check_every = 10
        for iteration in range(niter):
            means, covariances, mixing, loglike = expectation_maximization_step(means, covariances, mixing)
            if (iteration + 1) % check_every:
                continue
            loglike = float(loglike)
            if not np.isfinite(loglike):
                raise FloatingPointError(f'EM diverged at iteration {iteration}; raise covariance_floor')
            if abs(loglike - previous) < tol * max(1., abs(loglike)):
                break
            previous = loglike
        else:
            logger.info(f'EM stopped at the {niter}-iteration cap for {ncomponents} components '
                        f'(relative log-likelihood change {abs(loglike - previous) / max(1., abs(loglike)):.2e})')
        logger.info(f'fitted {ncomponents} components to {nsamples} samples in {iteration + 1} EM iterations')

        density = cls(weights=np.asarray(mixing), means=np.asarray(means), covariances=np.asarray(covariances),
                      names=names, floor_weight=floor_weight, loc=loc, scale=scale, limits=limits, **kwargs)
        density.attrs.update({'ncomponents': str(ncomponents), 'niter': str(iteration + 1),
                              'nsamples': str(nsamples), 'covariance_floor': str(covariance_floor),
                              'loglike': str(previous)})
        return density

    # ── evaluation ───────────────────────────────────────────────────────────

    def _log_prob_standardized(self, x):
        import jax
        import jax.numpy as jnp
        offset = jnp.asarray(x)[None] - jnp.asarray(self.means)
        precisions = jnp.linalg.inv(jnp.asarray(self.covariances))
        _, logdets = jnp.linalg.slogdet(jnp.asarray(self.covariances))
        mahalanobis = jnp.einsum('kd,kde,ke->k', offset, precisions, offset)
        logpdf = jax.scipy.special.logsumexp(
            -0.5 * mahalanobis - 0.5 * logdets - 0.5 * self.ndim * jnp.log(2 * jnp.pi) + jnp.log(jnp.asarray(self.weights)))
        if self.floor_weight > 0:
            logpdf = jnp.logaddexp(logpdf, jnp.log(self.floor_weight) + self._floor_log_prob(x))
        return logpdf

    def _floor_log_prob(self, x):
        """A broad Gaussian covering the fitted range: the tail floor, in standardized units."""
        import jax.numpy as jnp
        spread = jnp.maximum(3., jnp.max(jnp.abs(jnp.asarray(self.means)), axis=0) * 2 + 3.)
        residual = jnp.asarray(x) / spread
        return (-0.5 * jnp.sum(residual ** 2) - jnp.sum(jnp.log(spread))
                - 0.5 * self.ndim * jnp.log(2 * jnp.pi))

    def sample(self, key, shape):
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom
        ntotal = int(np.prod(shape))
        key, component_key, draw_key = jrandom.split(key, 3)
        component = jrandom.choice(component_key, self.ncomponents, (ntotal,), p=jnp.asarray(self.weights))
        cholesky = jnp.linalg.cholesky(jnp.asarray(self.covariances) + 1e-12 * jnp.eye(self.ndim))
        standard = jrandom.normal(draw_key, (ntotal, self.ndim))
        draws = jnp.asarray(self.means)[component] + jnp.einsum('nd,nde->ne', standard, cholesky[component])
        return (jnp.asarray(self.loc) + jnp.asarray(self.scale) * draws).reshape(tuple(shape) + (self.ndim,))

    def mean(self):
        mixture_mean = (np.asarray(self.weights)[:, None] * np.asarray(self.means)).sum(axis=0)
        return self.loc + self.scale * mixture_mean

    def covariance(self):
        """Law of total covariance: within-component spread plus between-component spread.

        Both terms carry the mixing weight -- dropping it on the within term over-counts the
        covariance by roughly ``ncomponents``, which is easy to miss because the density
        itself is unaffected.
        """
        means = np.asarray(self.means)
        mixing = np.asarray(self.weights)[:, None, None]
        mixture_mean = (np.asarray(self.weights)[:, None] * means).sum(axis=0)
        offset = means - mixture_mean
        within = (mixing * np.asarray(self.covariances)).sum(axis=0)
        between = (mixing * offset[:, :, None] * offset[:, None, :]).sum(axis=0)
        scale = np.asarray(self.scale)
        return scale[:, None] * (within + between) * scale[None, :]

    def arrays(self):
        return {'weights': self.weights, 'means': self.means, 'covariances': self.covariances,
                'floor_weight': np.asarray(self.floor_weight)}

    def _set_arrays(self, arrays):
        self.weights = arrays['weights']
        self.means = arrays['means']
        self.covariances = arrays['covariances'].reshape(len(self.weights), self.ndim, self.ndim)
        self.ncomponents = len(self.weights)
        self.floor_weight = float(arrays.get('floor_weight', 0.))
