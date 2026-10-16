"""Non-Gaussian likelihood for compressed (summary-statistic) observables.

The compressed likelihoods in :mod:`desilike.likelihoods.base` reduce a first-stage chain to
its mean and covariance and evaluate a Gaussian at the theory prediction.  That is a good
approximation for BAO distances and a poor one when the compressed parameters are non-linearly
degenerate -- ``df`` and ``dA`` in ShapeFit, whose joint posterior curves.  This module
replaces the mean/covariance reduction with a density learned from the chain itself:

.. math::

    \\ln\\mathcal{L}(\\theta) = \\ln q\\big(c_{\\rm th}(\\theta)\\big)
                              - \\ln\\pi_1\\big(c_{\\rm th}(\\theta)\\big)

with :math:`q` the fitted density (:mod:`desilike.densities`) and :math:`\\pi_1` the
*first-stage* prior on the compressed parameters.

Why the prior is divided out
----------------------------
The chain is a posterior, not a likelihood.  With :math:`L_1(c) = \\int \\mathrm{d}\\nu\\,
\\mathcal{L}(D_1|c,\\nu)\\pi(\\nu)` the first-stage likelihood marginalized over its nuisances,

.. math::

    p(c\\,|\\,D_1) = L_1(c)\\,\\pi_1(c)\\,/\\,Z
    \\quad\\Longrightarrow\\quad
    L_1(c) = Z\\,p(c\\,|\\,D_1)\\,/\\,\\pi_1(c).

So the effective likelihood is the chain density *divided by the prior it was sampled under*.
:math:`Z` does not depend on the second-stage parameters, so it drops out of the posterior
over :math:`\\theta` -- but it does not drop out of an evidence comparison, which is the one
number this approximation cannot be trusted for.  When :math:`\\pi_1` is a uniform box (it is,
for ``qpar``/``qper``/``df``/``dm``/``dA`` in
:class:`~desilike.theories.galaxy_clustering.template.ShapeFitSpectrum2Template`) the division
is a constant inside the box and a mask outside it, so ``limits`` alone is enough -- and the
mask is not a detail: the mean/covariance reduction leaks Gaussian mass past the prior box,
where the chain has no support at all.

Classes
-------
ObservablesDensityLikelihood
    Non-Gaussian likelihood combining one or more compressed observables with a learned density.
"""

import numpy as np
import jax.numpy as jnp
import lsstypes as types

from ..base import Likelihood
from ..densities import fit_density
from ..parameter import Parameter, Variable, VariableCollection


def _collect_observables(observables):
    """Assemble the joint lsstypes data tree for *observables*.

    Lifted out of :class:`~desilike.likelihoods.base.ObservablesGaussianLikelihood.__init__`
    (everything but the covariance handling) so the Gaussian and the density likelihoods share
    one implementation of it rather than two that drift apart.  Moving it there and importing
    it back is the intended final state; it lives here until that refactor is made.

    Returns
    -------
    observables : list
    data : lsstypes.ObservableTree
    flatdata : numpy.ndarray
    """
    if not isinstance(observables, (list, tuple)):
        observables = [observables]
    observables = list(observables)
    data = [observable.data for observable in observables]
    data = types.ObservableTree(data, observables=[observable.name for observable in observables])
    return observables, data, data.value()


class ObservablesDensityLikelihood(Likelihood):
    r"""Non-Gaussian likelihood for compressed observables, from a density learned on a chain.

    A drop-in replacement for :class:`~desilike.likelihoods.ObservablesGaussianLikelihood` on
    compressed observables: same ``observables``, same ``parameters`` order, no covariance.
    Where the Gaussian one computes :math:`-\tfrac12 r^\mathsf{T} C^{-1} r` with
    :math:`r = c_{\rm data} - c_{\rm th}`, this evaluates a fitted density at
    :math:`c_{\rm th}` alone -- the data are *inside* the density, having been used to fit it,
    so there is no residual and no covariance.  ``flatdata`` is still assembled (the mean of
    the chain, say) because ``ndata``, diagnostics and synthetic-data generation read it.

    This is a :class:`~desilike.base.Likelihood`, deliberately **not** a
    :class:`~desilike.base.GaussianLikelihood`.  See the Note below.

    Parameters
    ----------
    observables : Calculator or list of Calculator
        Compressed observables exposing ``flattheory`` and ``data``, e.g.
        :class:`~desilike.observables.galaxy_clustering.ShapeFitCompressionObservable`
        constructed with ``covariance=None``.
    density : Density, default=None
        An already-fitted density.  Mutually exclusive with *chain*; when neither is given the
        class raises.  Passing a fitted object is the recommended route -- fit it once, cache
        it with :meth:`~desilike.densities.Density.write`, and hand it over.
    chain : MCSamples or list of MCSamples, default=None
        Fit the density at ``build()`` time instead.  The fit is cached to *filename* and
        re-read on later builds when the recorded config still matches, because ``build()``
        re-runs ``__init__`` and ``__post_init__``.
    parameters : list of str, default=None
        The compressed parameter names, in the order the density's input vector is built.
        Defaults to the observables' own ``parameters`` concatenated, and is checked against
        ``density.names`` -- a silent order mismatch between the chain and ``flattheory`` would
        otherwise produce a confidently wrong posterior.
    kind : str, default='gmm'
        Backend when fitting from *chain*: ``'gmm'``, ``'gaussian'`` or ``'flow'``.
    priors : VariableCollection, dict or Calculator, default=None
        The **first-stage** priors on *parameters*, whose log-pdf is subtracted (see the module
        docstring).  Normally omitted: they are read off *chain* (whose Parameters keep them
        through an HDF5 round trip), or off the fitted *density*'s recorded attrs, in that order.
        Pass them only to override -- e.g. ``get_params(first_stage_likelihood)``.
    limits : array, shape (nparameters, 2), default=None
        Support box, read off *priors* when given.  Outside it the likelihood is ``-inf``
        (or the floor below, when one is set).
    filename : str, default=None
        Where to cache the fitted density.
    tail_floor : float, default=1e-4
        Weight of a broad Gaussian added in quadrature to the density,
        ``logaddexp(log q, log(tail_floor) + log N_broad)``.  The chain under-samples its own
        tails, so a fitted density there is noise; without a floor that noise becomes a wall a
        sampler cannot cross and a NaN for a gradient-based kernel (desilike sets
        ``EQX_ON_ERROR=nan``, so the NaN propagates rather than raising).  Set to 0 to disable,
        and check the posterior does not move when it is varied by a decade.
    divide_prior : bool, default=True
        Subtract ``priors``' log-pdf.  Turn off only if the density was fitted to
        prior-divided samples already.
    **fit_options
        Forwarded to :func:`~desilike.densities.fit_density` when fitting from *chain*
        (``ncomponents``, ``knots``, ``select_from``, ``seed``, ...).

    Notes
    -----
    Why not inherit from :class:`~desilike.base.GaussianLikelihood`
        :func:`~desilike.base.Posterior` dispatches on ``isinstance(likelihood,
        GaussianLikelihood)``.  A Gaussian leaf with solved parameters (``derived='marg'`` or
        ``'best'``) is marginalised *analytically*: ``Posterior`` reads its ``precision`` and
        ``flatdata``, builds a graph over its ``flattheory``, ``jax.linearize``\\ s it in the
        solved parameters and applies a Schur complement -- exact only for a likelihood that is
        quadratic in the residual.  On a fitted density that is simply wrong, and it fails
        silently, returning a plausible but biased posterior and evidence.  The non-Gaussian
        branch is the safe one: its ``logpdf`` is added directly, and a non-Gaussian component
        that *does* touch a solved parameter raises an explicit ``ValueError`` at build time.
        Inheriting from the Gaussian class to reuse its data plumbing therefore buys nothing
        that ``_collect_observables`` does not, at the cost of being mis-classified.

    What the flow has to do with a Gaussian
        A flow *is* a nonlinear whitening transform, so the intuition is right in a specific
        sense.  Fitted so that :math:`z = T(c)` is standard normal,

        .. math:: \\ln q(c) = -\\tfrac12 \\lVert T(c)\\rVert^2 - \\tfrac{d}{2}\\ln 2\\pi
                    + \\ln\\left|\\det \\partial T/\\partial c\\right|(c),

        i.e. the usual :math:`\\chi^2` with :math:`L^{-1}(c - c_{\rm data})` replaced by a
        learned :math:`T`, **plus a log-determinant with no Gaussian analogue**.  That term is
        what accounts for the volume distortion, and dropping it biases exactly the curved
        directions one built the flow for.  No Gaussian *samples* are involved in evaluation:
        ``log_prob`` pushes the point forward through :math:`T`; drawing from the base
        distribution is only needed to sample the flow, which the likelihood never does.

    Examples
    --------
    Fit once and cache::

        chain = MCSamples.read('_tests/chain_fs_shapefit_Ap/samples_1.h5').remove_burnin(0.3)
        parameters = ['qpar', 'qper', 'df', 'dm', 'dA']
        density = fit_density(chain, parameters, kind='gmm',
                              select_from=[{'ncomponents': k} for k in (8, 16, 32, 64)],
                              priors=first_stage_params, filename='shapefit_density.h5')
        observable = ShapeFitCompressionObservable(data=density.mean(), covariance=None,
                                                   parameters=parameters, z=z)
        likelihood = ObservablesDensityLikelihood(observables=[observable], density=density)
        sampler = Sampler(build(Posterior(likelihood)), kernel=Zeus(), rng=42)
    """

    def __init__(self, observables, density=None, chain=None, parameters=None, kind='gmm',
                 priors=None, limits=None, filename=None, tail_floor=1e-4, divide_prior=True,
                 **fit_options):
        if density is None and chain is None:
            raise ValueError('provide either a fitted density= or a chain= to fit one to')
        if density is not None and chain is not None:
            raise ValueError('provide density= or chain=, not both')
        # The observables are the only nodes; everything else is configuration or a fitted
        # (non-Node) object, so it is kept private and `_trace_graph` does not scan it.
        self.observables, self.data, flatdata = _collect_observables(observables)
        if parameters is None:
            parameters = [name for observable in self.observables for name in observable.parameters]
        self.parameters = [name if isinstance(name, str) else name.name for name in parameters]
        if len(self.parameters) != flatdata.size:
            raise ValueError(f'{len(self.parameters)} parameters but the observables hold '
                             f'{flatdata.size} data points; repeated parameter names (several '
                             f'measurements of one quantity) are not supported by a density')
        # A Variable, not a Parameter, for the reason GaussianLikelihood's docstring gives:
        # it is a node the graph fills and can be overridden by name (an Asimov vector, a mock),
        # while staying out of `select(varied=True)` and out of every chain.
        self.flatdata = Variable(f'{type(self).__name__}.flatdata', value=jnp.asarray(flatdata))
        self._flatdata = flatdata
        self._density = density
        self._chain = chain
        self._kind = str(kind)
        self._filename = filename
        self._fit_options = dict(fit_options)
        self._tail_floor = float(tail_floor)
        self._divide_prior = bool(divide_prior)
        # The first-stage priors drive both the support box and the prior division, and they are
        # read rather than asked for wherever possible: the chain's Parameters carry them through
        # an HDF5 round trip; failing that, the fitted density records what it was fitted with;
        # failing that, they must be given explicitly.
        from ..densities.base import resolve_priors, _limits_from_priors
        self._priors = resolve_priors(chain, self.parameters, priors)
        if self._priors is None and density is not None:
            recovered = density.priors()
            if recovered is not None:
                self._priors = VariableCollection(
                    [Parameter(name, prior=prior) if prior is not None else Parameter(name)
                     for name, prior in recovered.items()])
        self._limits = None
        if limits is not None:
            self._limits = np.asarray(limits, dtype='f8').reshape(len(self.parameters), 2)
        elif self._priors is not None:
            self._limits = _limits_from_priors(self._priors, self.parameters)
        elif density is not None and density.limits is not None:
            self._limits = np.asarray(density.limits, dtype='f8')

    def __post_init__(self, *args, **kwargs):
        """Fit or load the density.  Non-node setup, and idempotent: this runs at every build."""
        if self._density is None:
            self._density = fit_density(self._chain, self.parameters, kind=self._kind,
                                        limits=self._limits, priors=self._priors,
                                        filename=self._filename, **self._fit_options)
        if list(self._density.names) != list(self.parameters):
            raise ValueError(f'the density is over {list(self._density.names)} but the '
                             f'observables predict {list(self.parameters)}; a mismatch here '
                             f'would silently compare the wrong quantities')
        if self._density.ndim != self._flatdata.size:
            raise ValueError(f'the density has {self._density.ndim} dimensions but the '
                             f'observables hold {self._flatdata.size} data points')
        # Moment-matched covariance, for diagnostics and proposals only -- not what the
        # likelihood evaluates, which is the point of this class.  Kept here rather than
        # written onto the observables: theirs is an lsstypes CovarianceMatrix, and replacing
        # it with a desilike Covariance would break anything that matches it by observable.
        self.covariance = self._density.to_covariance()
        # The priors belong to the *first stage* and are not nodes of this graph, so their
        # Parameter.value is meaningless here; what matters is the prior as a function, bound
        # to the column of flattheory it applies to.
        self._prior_terms = []
        if self._divide_prior and self._priors is not None:
            for index, name in enumerate(self.parameters):
                prior = getattr(self._priors[name], 'prior', None)
                if prior is None:
                    continue
                shape = getattr(self._priors[name], 'shape', ()) or ()
                if shape:
                    raise ValueError(f'prior division supports scalar compressed parameters only; '
                                     f'{name!r} has shape {shape}')
                self._prior_terms.append((index, prior))
        self._broad_precision = None
        if self._tail_floor > 0:
            broad = np.asarray(self._density.covariance(), dtype='f8') * 25.  # 5x the width per axis
            self._broad_center = jnp.asarray(self._density.mean())
            self._broad_logdet = float(np.linalg.slogdet(broad)[1])
            self._broad_precision = jnp.linalg.inv(jnp.asarray(broad))

    @property
    def density(self):
        """The fitted :class:`~desilike.densities.Density` this likelihood evaluates."""
        return self._density

    @property
    def ndata(self):
        return self._flatdata.size

    def __call__(self):
        self.flattheory = jnp.concatenate([jnp.ravel(jnp.asarray(observable.flattheory))
                                           for observable in self.observables])
        self.flatdata = jnp.concatenate([jnp.ravel(jnp.asarray(observable.flatdata))
                                         for observable in self.observables])
        logpdf = self._density.log_prob(self.flattheory)
        for index, prior in self._prior_terms:
            # ParameterPrior.logpdf is relative to its centre, so this drops a
            # parameter-independent normalization; harmless for the posterior over theta, and
            # the same convention the Prior calculator uses.
            logpdf = logpdf - prior.logpdf(self.flattheory[index])
        if self._broad_precision is not None:
            residual = self.flattheory - self._broad_center
            broad = (-0.5 * residual @ (self._broad_precision @ residual)
                     - 0.5 * self._broad_logdet - 0.5 * self._density.ndim * jnp.log(2 * jnp.pi))
            logpdf = jnp.logaddexp(logpdf, jnp.log(self._tail_floor) + broad)
        self.logpdf = logpdf
        return self.logpdf

    def tree_flatten(self):
        return [self.logpdf, self.flattheory], None

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.logpdf, obj.flattheory = children
        return obj

    @classmethod
    def install(cls, installer):
        """Install the optional extra a ``kind='flow'`` density needs."""
        installer.pip('flowjax')
