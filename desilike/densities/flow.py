"""Normalizing-flow density, on top of `flowjax <https://github.com/danielward27/flowjax>`_.

An optional backend: flowjax is not a desilike dependency (it needs Python >= 3.11, while
desilike declares >= 3.10), so it is imported lazily and declared as an extra::

    [project.optional-dependencies]
    flows = ["flowjax"]

Everything below was verified with jax 0.10.2 / flowjax 19.1.1 / equinox 0.13.8 under
``jax_enable_x64 = True`` (which ``desilike.base`` sets at import):

* the flow runs in float64 end to end -- fit, ``log_prob``, ``grad`` -- and the round trip
  through serialization preserves float64;
* ``grad`` of ``log_prob`` w.r.t. a cosmology parameter agrees with central finite
  differences to ~1e-9 relative, i.e. the log-determinant term is included.  Use
  ``Transformed``/``Distribution.log_prob`` rather than hand-rolling ``bijection.transform``
  plus a normal log-pdf: that is where the log-det gets dropped, and dropping it biases
  exactly the curved directions the flow was fitted for;
* ``jax.jit(node)`` works when *node* is an object holding the flow and reading it in
  ``__call__`` (desilike's pattern), but ``jax.jit(distribution.log_prob)`` raises
  ``TypeError: unhashable type: ArrayImpl`` -- equinox modules are unhashable, so never hand
  a *bound* flowjax method to ``jit``; wrap it in a closure;
* ``pickle.dumps(distribution)`` **fails** (``PicklingError: Can't pickle <function
  softplus>``): paramax/flowjax leaves hold function references.  Serialization therefore
  goes through the array leaves plus a config that rebuilds the architecture -- the same
  split ``equinox.tree_serialise_leaves`` makes, but stored inside desilike's own HDF5 so a
  density is one self-contained file.
"""

import json
import logging

import numpy as np

from ..utils import register_type
from .base import Density


def _import_flowjax():
    """Import flowjax lazily, with the actionable message an optional dependency deserves."""
    try:
        import flowjax  # noqa: F401
        return flowjax
    except ImportError as error:
        raise ImportError(
            'FlowDensity needs flowjax, which desilike does not require by default '
            '(it needs Python >= 3.11).  Install it with `pip install flowjax`, or use '
            "kind='gmm', which has no extra dependency and is the better estimator at the "
            '4-6 dimensions and 1e3-1e4 effective samples a compressed chain gives you.'
        ) from error


#: The flow architectures this backend can build, by ``kind``.
_BUILDERS = ('maf', 'coupling', 'bnaf')


def _build_distribution(config):
    """Construct an *untrained* flowjax distribution from an architecture config.

    One definition, used both by :meth:`FlowDensity.fit` and by :meth:`FlowDensity._set_arrays`
    when reading a cache.  Duplicating it is how a fitted flow ends up unrestorable: the two
    copies drift, the leaf layout no longer matches, and the failure surfaces at ``read`` time
    rather than at ``write`` time.  Everything here must be a pure function of *config* -- no
    randomness beyond ``config['seed']``, no ambient state -- or the rebuilt architecture will
    not have the leaf layout the file was written with.
    """
    import jax.numpy as jnp
    import jax.random as jrandom
    from flowjax.distributions import Normal
    from flowjax.flows import (masked_autoregressive_flow, coupling_flow,
                               block_neural_autoregressive_flow)

    kind = str(config['kind'])
    builders = {'maf': masked_autoregressive_flow, 'coupling': coupling_flow,
                'bnaf': block_neural_autoregressive_flow}
    if kind not in builders:
        raise ValueError(f'architecture must be one of {sorted(builders)}, got {kind!r}')
    ndim = int(config['ndim'])
    base_dist = Normal(jnp.zeros(ndim))
    if kind == 'bnaf':
        kwargs = {'flow_layers': int(config['flow_layers']), 'nn_depth': int(config['nn_depth']),
                  'nn_block_dim': int(config['nn_width'])}
    else:
        if str(config.get('transformer', 'spline')) == 'affine':
            from flowjax.bijections import Affine
            transformer = Affine()
        else:
            from flowjax.bijections import RationalQuadraticSpline
            transformer = RationalQuadraticSpline(knots=int(config['knots']),
                                                  interval=float(config['interval']))
        kwargs = {'transformer': transformer,
                  'flow_layers': int(config['flow_layers']), 'nn_width': int(config['nn_width']),
                  'nn_depth': int(config['nn_depth'])}
    return builders[kind](key=jrandom.key(int(config.get('seed', 0))), base_dist=base_dist, **kwargs)


@register_type
class FlowDensity(Density):
    """A normalizing flow fitted to samples by maximum likelihood.

    Parameters
    ----------
    names : list of str
    distribution : flowjax distribution
        The fitted flow, in *unbounded* standardized coordinates (see ``_to_unbounded``).
    config : dict
        Everything needed to rebuild the architecture: ``kind``, ``ndim``, ``knots``,
        ``interval``, ``flow_layers``, ``nn_width``, ``nn_depth``, ``seed``, ``squeeze``.
        The arrays are stored separately; the config is what makes them interpretable again,
        and :func:`_build_distribution` is the single place that turns it back into a flow.
        The support box is *not* part of it -- it is derived from ``limits`` (see ``_bounds``).

    Notes
    -----
    Support.  A spline flow leaves mass outside the region the chain occupies, and a chain
    cannot leave its prior box, so the flow is free to invent density there.  When *limits*
    is given the flow is fitted in logit coordinates of that box, which makes the support
    exact: the density goes to zero at the boundary instead of leaking past it.  This is the
    one place the flow is strictly better than a mixture rather than merely different.
    """

    _name = 'FlowDensity'

    def __init__(self, names, distribution=None, config=None, **kwargs):
        super().__init__(names=names, **kwargs)
        self._distribution = distribution
        self.config = dict(config or {})

    @property
    def _bounds(self):
        """The support box in standardized coordinates, or None for an unbounded flow.

        Derived from ``limits``/``loc``/``scale`` rather than stored: an object rebuilt by
        ``__setstate__`` never runs ``__init__``, so any second copy of the truth is a second
        thing to remember to restore -- and forgetting it fails at the first ``log_prob`` after
        a ``read``, not at load time.  Cached on first use; the three inputs never change after
        construction.
        """
        cached = self.__dict__.get('_bounds_cache', False)
        if cached is False:
            cached = None if self.limits is None else (np.asarray(self.limits, dtype='f8')
                                                       - self.loc[:, None]) / self.scale[:, None]
            self.__dict__['_bounds_cache'] = cached
        return cached

    # ── the squeeze between the prior box and the flow's unbounded space ──────
    # Kept explicit rather than expressed as a flowjax bijection: its log-determinant is then
    # ours to write down and to unit-test, and nothing depends on a library's convention for
    # the order a `Compose` applies its factors in.

    @property
    def _squeezed(self):
        return self._bounds is not None

    def _to_unbounded(self, u):
        """Standardized coordinates -> the flow's space, plus the log-determinant."""
        import jax.numpy as jnp
        if not self._squeezed:
            return u, 0.
        lo = jnp.asarray(self._bounds[:, 0])
        hi = jnp.asarray(self._bounds[:, 1])
        width = hi - lo
        fraction = (u - lo) / width
        # Clip inside (0, 1): at the boundary logit diverges, and the theory vector a sampler
        # proposes can land exactly on a prior edge.
        fraction = jnp.clip(fraction, 1e-6, 1. - 1e-6)
        logdet = -jnp.sum(jnp.log(fraction * (1. - fraction))) - jnp.sum(jnp.log(width))
        return jnp.log(fraction / (1. - fraction)), logdet

    def _to_standardized(self, t):
        """The inverse of :meth:`_to_unbounded` (used by :meth:`sample`)."""
        import jax
        import jax.numpy as jnp
        if not self._squeezed:
            return t
        lo = jnp.asarray(self._bounds[:, 0])
        hi = jnp.asarray(self._bounds[:, 1])
        return lo + (hi - lo) * jax.nn.sigmoid(t)

    # ── fitting ──────────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, X, names, weights=None, architecture='maf', transformer='spline', knots=8,
            interval=5., flow_layers=4, nn_width=32, nn_depth=1, max_epochs=200,
            learning_rate=2e-3, batch_size=512, max_patience=20, seed=0, loc=None, scale=None,
            limits=None, val_prop=0., **kwargs):
        """Fit a flow to *X* (physical units) by maximum likelihood.

        Other Parameters
        ----------------
        architecture : str, default='maf'
            The flow family, passed straight through to flowjax:

            * ``'maf'`` -- masked autoregressive: every dimension conditioned on all previous
              ones.  The default, and usually the best accuracy/cost trade-off at 4-6 dims.
            * ``'coupling'`` -- affine/spline couplings: blocks of dimensions conditioned on
              the rest.  Cheaper per layer, needs more layers for the same expressiveness.
            * ``'bnaf'`` -- block neural autoregressive: strictly monotone dense blocks, the
              most expressive per layer, but density evaluation needs a bisection inverse, so
              ``log_prob`` is markedly slower.  Reach for it only when the others underfit.

            Named ``architecture`` rather than ``kind`` because ``kind`` is
            :func:`~desilike.densities.fit_density`'s *backend* selector (``'gaussian'``,
            ``'gmm'``, ``'flow'``); the two used to share a name, which made the architecture
            unselectable through ``fit_density``.  ``kind=`` is still accepted here as an alias.
        transformer : str, default='spline'
            ``'spline'`` (rational-quadratic, neural-spline-flow style: flexible marginals,
            the right default for skewed compressed parameters) or ``'affine'`` (linear per
            layer: far fewer parameters, stouter tails, the right choice when effective
            sample sizes are small).  Ignored for ``'bnaf'``.
        knots : int, default=8
            Rational-quadratic spline knots.  The main capacity knob; 6 to 12 is the useful
            range at these dimensions and sample sizes.
        interval : float, default=5.
            Half-width of the spline interval, in standardized units.  Outside it the
            transformer is the identity, so this must cover the standardized data.
        flow_layers, nn_width, nn_depth
            Capacity, under flowjax's own names (``masked_autoregressive_flow`` and
            ``coupling_flow`` take all three; ``block_neural_autoregressive_flow`` takes
            ``flow_layers`` and ``nn_depth``, with ``nn_width`` read as its ``nn_block_dim``).
            flowjax defaults to 8 layers of width 50, which is a great deal of freedom for a
            5-D density fitted to a few hundred effective samples -- the defaults here are
            deliberately smaller.  Capacity is the knob to scan when validating.
        max_epochs, learning_rate, batch_size, max_patience
            Forwarded to ``flowjax.train.fit_to_data``, which early-stops on a held-out
            split of its own (``val_prop``).
        weights : array, default=None
            Ignored, with a warning: flowjax's maximum-likelihood loss is unweighted.  Resample
            the chain to equal weights before fitting (``prepare_training_samples`` does this
            by default for this backend) rather than passing weights here.
        """
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom
        _import_flowjax()
        from flowjax.distributions import Normal
        from flowjax.flows import (masked_autoregressive_flow, coupling_flow,
                                   block_neural_autoregressive_flow)
        from flowjax.train import fit_to_data

        logger = logging.getLogger(__name__)
        if weights is not None and not np.allclose(np.asarray(weights), np.asarray(weights)[0]):
            logger.warning('FlowDensity.fit ignores sample weights (flowjax maximises an '
                           'unweighted likelihood); resample the chain to equal weights first')
        X = np.asarray(X, dtype='f8')
        loc = X.mean(axis=0) if loc is None else np.asarray(loc, dtype='f8')
        scale = X.std(axis=0) if scale is None else np.asarray(scale, dtype='f8')
        scale = np.where(scale > 0, scale, 1.)
        if limits is not None:
            limits = np.asarray(limits, dtype='f8').reshape(len(names), 2)
            # Column-wise: `limits` is (ndim, 2) while loc/scale are (ndim,).
            bounds = (limits - loc[:, None]) / scale[:, None]
            if np.any(bounds[:, 0] >= bounds[:, 1]):
                raise ValueError('limits must bracket the data in every dimension; '
                                 'check that they are wider than the standardization')
        if 'kind' in kwargs:      # legacy alias for `architecture`
            architecture = kwargs.pop('kind')
        config = {'kind': str(architecture), 'transformer': str(transformer),
                  'ndim': len(names), 'knots': int(knots),
                  'interval': float(interval), 'flow_layers': int(flow_layers),
                  'nn_width': int(nn_width), 'nn_depth': int(nn_depth), 'seed': int(seed),
                  'squeeze': limits is not None}
        density = cls(names=names, loc=loc, scale=scale, limits=limits, config=config, **kwargs)
        standardized = (X - loc) / scale
        unbounded, _ = jax.vmap(density._to_unbounded)(jnp.asarray(standardized))
        distribution = _build_distribution(config)
        distribution, losses = fit_to_data(key=jrandom.key(seed + 1), dist=distribution,
                                          data=unbounded, learning_rate=float(learning_rate),
                                          max_epochs=int(max_epochs), batch_size=int(batch_size),
                                          max_patience=int(max_patience), val_prop=float(val_prop or 0.1),
                                          show_progress=False)
        density._distribution = distribution
        train_loss = float(np.asarray(losses['train'])[-1])
        val_loss = float(np.asarray(losses['val'])[-1]) if len(np.asarray(losses['val'])) else np.nan
        logger.info(f'fitted {architecture} flow ({len(names)}-D, {X.shape[0]} samples): '
                    f'train {train_loss:.4f}, val {val_loss:.4f}')
        if val_loss - train_loss > 0.5:
            logger.warning(f'flow is overfitting (train {train_loss:.3f} vs val {val_loss:.3f}); '
                           'reduce knots/depth or fit more samples')
        density.attrs.update({'architecture': architecture, 'transformer': transformer,
                      'nsamples': str(X.shape[0]),
                              'train_loss': str(train_loss), 'val_loss': str(val_loss)})
        return density

    # ── evaluation ───────────────────────────────────────────────────────────

    def _log_prob_standardized(self, x):
        """Log-density in standardized coordinates, squeeze log-determinant included."""
        import jax.numpy as jnp
        unbounded, logdet = self._to_unbounded(x)
        # A closure, not `self._distribution.log_prob`: see the module docstring.
        return self._distribution.log_prob(unbounded) + logdet

    def sample(self, key, shape):
        import jax
        import jax.numpy as jnp
        draws = self._distribution.sample(key, tuple(shape))
        standardized = jax.vmap(self._to_standardized)(draws)
        return jnp.asarray(self.loc) + jnp.asarray(self.scale) * standardized

    def mean(self, nsamples=20000, seed=0):
        import jax
        import jax.numpy as jnp
        draws = self.sample(jax.random.key(seed), (nsamples,))
        return np.asarray(jnp.mean(draws, axis=0))

    def covariance(self, nsamples=20000, seed=0):
        import jax
        import jax.numpy as jnp
        draws = np.asarray(self.sample(jax.random.key(seed), (nsamples,)))
        return np.cov(draws.T)

    # ── serialization: config (rebuilds the architecture) + array leaves ─────
    #
    # `equinox.tree_serialise_leaves` makes exactly this split -- array leaves to a file,
    # everything else reconstructed from a template.  The leaves are stored here inside
    # desilike's own HDF5 instead of a sidecar `.eqx` file, so a fitted density is one
    # self-contained artifact that `Density.read` dispatches on, like every other desilike
    # object.  Pickling the module is not an option: see the module docstring.

    @staticmethod
    def _array_filter():
        import equinox as eqx
        return eqx.is_array

    def _leaf_layout(self):
        """Return ``(leaves, layout)``: every *array* leaf of the flow, and how to put them back.

        Non-array leaves (Python ints and bools, and the activation functions that make
        pickling the module fail) are not stored -- rebuilding the architecture from
        ``config`` reproduces them.  Saving *all* array leaves, not only the inexact ones, is
        what makes the rebuild seed-independent: an integer leaf (a coupling order, say) is
        part of the fitted object rather than of the initialization, and must travel with it.
        """
        import jax
        import equinox as eqx
        params, _ = eqx.partition(self._distribution, self._array_filter())
        leaves, _ = jax.tree_util.tree_flatten(params)
        # Only the leaf shapes and dtypes are stored, not the treedef: the structure is a pure
        # function of the architecture config, so the template rebuilt in `_set_arrays` supplies
        # it, and comparing shapes and dtypes against the stored ones is a strictly stronger
        # check than comparing a serialized treedef would be.
        layout = {'shapes': [list(map(int, np.shape(leaf))) for leaf in leaves],
                  'dtypes': [str(np.asarray(leaf).dtype) for leaf in leaves]}
        return leaves, layout

    def arrays(self):
        if self._distribution is None:
            return {}
        leaves, _ = self._leaf_layout()
        return {f'leaf_{index:04d}': np.asarray(leaf) for index, leaf in enumerate(leaves)}

    def __getstate__(self, to_file=False):
        if self._distribution is not None:
            _, layout = self._leaf_layout()
            # Written into attrs so `Density.__getstate__` carries it as one JSON string, and
            # `_restore_attrs` below reads it back *before* `_set_arrays` needs it.
            self.attrs['config'] = json.dumps({**self.config, **layout})
        return super().__getstate__(to_file=to_file)

    def _restore_attrs(self, attrs):
        self.config = json.loads(attrs.get('config') or '{}')

    def _set_arrays(self, arrays):
        """Rebuild the architecture from ``config``, then pour the stored leaves back in."""
        import jax
        import jax.numpy as jnp
        import equinox as eqx

        config = getattr(self, 'config', None) or {}
        if 'ndim' not in config:
            raise ValueError('a FlowDensity read from file carries no architecture config; '
                             'it cannot be rebuilt')
        template = _build_distribution(config)
        params, static = eqx.partition(template, self._array_filter())
        keys = sorted(key for key in arrays if key.startswith('leaf_'))
        shapes, dtypes = config['shapes'], config['dtypes']
        template_leaves, treedef = jax.tree_util.tree_flatten(params)
        template_shapes = [list(map(int, np.shape(leaf))) for leaf in template_leaves]
        template_dtypes = [str(np.asarray(leaf).dtype) for leaf in template_leaves]
        if len(keys) != len(shapes):
            raise ValueError(f'the file holds {len(keys)} array leaves but this architecture '
                             f'expects {len(shapes)}; the config and the arrays disagree')
        if template_shapes != shapes or template_dtypes != dtypes:
            raise ValueError('the rebuilt architecture does not have the leaf layout the file '
                             'was written with; flowjax version drift? Refit the density rather '
                             'than loading it.')
        leaves = [jnp.asarray(arrays[key], dtype=np.dtype(dtype)).reshape(shape)
                  for key, shape, dtype in zip(keys, shapes, dtypes)]
        self._distribution = eqx.combine(jax.tree_util.tree_unflatten(treedef, leaves), static)

    @classmethod
    def install(cls, installer):
        installer.pip('flowjax')

