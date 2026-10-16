"""Tests for learned densities and the non-Gaussian compressed likelihood.

Layered, cheapest first.  Layers 1-2 need nothing but numpy/jax and run on every Python
version in the CI matrix; the flowjax backend is skipped where it is unavailable (it needs
Python >= 3.11, and desilike declares >= 3.10).  Layer 3 needs the real desilike graph, and
layer 4 -- the science validation -- is a notebook, not CI; see ``docs/dev/densities.rst``.

The invariant tests are parametrized over backends on purpose: a density is a contract, and
a new backend should be able to fail these rather than have its own weaker set.
"""

import os
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from desilike.densities import fit_density, GaussianDensity, GaussianMixtureDensity
from desilike.densities.base import Density, prepare_training_samples, chain_to_array


# Sizes are deliberately modest: this file runs on every pull request, across the whole CI
# Python matrix, on shared runners.  Bump them for a nightly run with
# DESILIKE_TEST_DENSITY_SCALE=4 -- the tolerances below are set for the default size.
SCALE_FACTOR = int(os.environ.get('DESILIKE_TEST_DENSITY_SCALE', '1'))
N_TRAIN, N_TEST = 1500 * SCALE_FACTOR, 600 * SCALE_FACTOR

NAMES = ['qpar', 'qper', 'df', 'dm', 'dA']
NDIM = len(NAMES)
LOC = np.array([1., 1., 1., 0., 1.])
SCALE = np.array([0.010, 0.012, 0.020, 0.030, 0.018])
# The flow backend is opt-in: fitting one is slow and memory-hungry enough that it does not
# belong on every commit.  CI runs the fast set per pull request and this one nightly, with
# DESILIKE_TEST_FLOWS=1.  flowjax also needs Python >= 3.11, and the CI matrix starts at 3.10.
BACKENDS = ['gaussian', 'gmm']
if os.environ.get('DESILIKE_TEST_FLOWS', '0') == '1' and sys.version_info >= (3, 11):
    try:
        import flowjax  # noqa: F401
        BACKENDS.append('flow')
    except ImportError:
        pass


# ── fixtures: a chain whose density is known EXACTLY ──────────────────────────
#
# A standard normal pushed through a triangular (hence exactly invertible, known log-det)
# autoregressive map, then rescaled to ShapeFit-like units.  `x2 = z2 + 0.8 z1**2` is the
# curved degeneracy that stands in for df-dA, and `x3`/`x5` carry the z1-dependent width
# that makes the distribution heteroscedastic.  Because the density is analytic, every
# accuracy statement below is a measurement against truth rather than against another fit.

def _forward(z):
    z1, z2, z3, z4, z5 = z
    x1 = z1
    x2 = z2 + 0.8 * x1 ** 2
    x3 = z3 * np.exp(0.5 * x1) + 0.4 * x2 ** 2
    x4 = z4 + 0.7 * np.tanh(x2 * x3)
    x5 = z5 * np.exp(-0.4 * x1) + 0.3 * x3
    return np.array([x1, x2, x3, x4, x5])


def _inverse(x):
    x1, x2, x3, x4, x5 = x
    return np.array([x1, x2 - 0.8 * x1 ** 2, (x3 - 0.4 * x2 ** 2) * np.exp(-0.5 * x1),
                     x4 - 0.7 * np.tanh(x2 * x3), (x5 - 0.3 * x3) * np.exp(0.4 * x1)])


def _physical(z):
    return LOC + SCALE * _forward(z)


def true_logpdf(x):
    """Exact log-density of the toy, in physical units -- the floor nothing can beat."""
    x = np.asarray(x, dtype='f8')
    z = _inverse((x - LOC) / SCALE)
    latent = -0.5 * z @ z - 0.5 * NDIM * np.log(2 * np.pi)
    # log|det d(physical)/d(z)| = sum(log SCALE) + the triangular map's own log-det
    squeeze = -np.sum(np.log(SCALE)) - 0.1 * (x[0] - LOC[0]) / SCALE[0]
    return latent + squeeze


def _toy_samples(nsamples, seed):
    rng = np.random.default_rng(seed)
    return np.array([_physical(z) for z in rng.normal(size=(nsamples, NDIM))])


@pytest.fixture(scope='module')
def toy():
    """Train/test split of the exactly-known toy, plus a support box that brackets it."""
    train, test = _toy_samples(N_TRAIN, 11), _toy_samples(N_TEST, 12)
    pad = 0.25 * np.ptp(train, axis=0)
    limits = np.column_stack([train.min(axis=0) - pad, train.max(axis=0) + pad])
    return dict(train=train, test=test, limits=limits,
                true_nll=-float(np.mean([true_logpdf(x) for x in test])))


@pytest.fixture(scope='module')
def fitted(toy):
    """One fitted density per backend, model-selected on held-out data."""
    out = {}
    for kind in BACKENDS:
        options = dict(select_from=[{'ncomponents': k} for k in (4, 12)]) if kind == 'gmm' \
            else dict(select_from=[{'knots': k, 'max_epochs': 40} for k in (6,)]) if kind == 'flow' else {}
        out[kind] = fit_density(toy['train'], NAMES, kind=kind, limits=toy['limits'],
                                val_prop=0.25, seed=3, verbose=False, **options)
    return out


def _density(kind, fitted):
    if kind == 'flow':
        pytest.importorskip('flowjax')
    return fitted[kind]


# ── layer 0: sample preparation ──────────────────────────────────────────────

def test_chain_to_array_order_and_weights():
    """Columns come out in the order asked for, not the order the chain happens to store."""
    from desilike.samples.chain import MCSamples
    from desilike.parameter import Parameter
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, NDIM))
    chain = MCSamples([Parameter(name, value=X[:, index]) for index, name in enumerate(NAMES)])
    got, weights, names = chain_to_array(chain, ['dA', 'qpar'])
    assert names == ['dA', 'qpar']
    np.testing.assert_allclose(got[:, 0], X[:, 4])
    np.testing.assert_allclose(got[:, 1], X[:, 0])
    assert weights.shape == (50,)

    shuffled, _, _ = chain_to_array(chain, ['qpar', 'dA'])
    assert not np.allclose(shuffled, got), 'column order must follow `parameters`'


def test_chain_to_array_rejects_unknown_parameter():
    from desilike.samples.chain import MCSamples
    from desilike.parameter import Parameter
    chain = MCSamples([Parameter('qpar', value=np.zeros(10))])
    with pytest.raises(ValueError, match='no parameter'):
        chain_to_array(chain, ['qpar', 'nope'])


def test_prepare_training_samples_is_deterministic(toy):
    """Same seed -> same training array.  A fit that is not reproducible cannot be cached."""
    first, _, _, _ = prepare_training_samples(toy['train'], NAMES, thin=None, nsamples=500, rng=7)
    second, _, _, _ = prepare_training_samples(toy['train'], NAMES, thin=None, nsamples=500, rng=7)
    np.testing.assert_array_equal(first, second)


def test_prepare_training_samples_reweights():
    """Non-uniform weights become an equally weighted set, of about the Kish ESS size."""
    from desilike.samples.chain import MCSamples
    from desilike.parameter import Parameter
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 2))
    chain = MCSamples([Parameter(name, value=X[:, index]) for index, name in enumerate(['a', 'b'])],
                      weight=rng.uniform(0.01, 1., 2000))
    got, weights, _, info = prepare_training_samples(chain, ['a', 'b'], thin=None, weighted='resample', rng=0)
    assert weights is None or np.allclose(weights, weights[0])
    assert 'kish_ess' in info
    assert got.shape[0] <= 2000


# ── layer 1: the density contract, per backend ───────────────────────────────

@pytest.mark.parametrize('kind', BACKENDS)
def test_log_prob_is_finite_float64_and_jittable(kind, fitted, toy):
    density = _density(kind, fitted)
    x = jnp.asarray(toy['train'][0])
    value = density.log_prob(x)
    assert value.shape == ()
    assert value.dtype == jnp.float64, 'desilike runs with jax_enable_x64; a float32 density loses precision'
    assert np.isfinite(float(value))
    # Never jit a bound method of an equinox module: it is unhashable.  A closure is fine,
    # and is what a Calculator node does.
    assert float(jax.jit(lambda point: density.log_prob(point))(x)) == pytest.approx(float(value), abs=1e-12)


# ── normalization ─────────────────────────────────────────────────────────────
#
# Checked by quadrature rather than importance sampling.  The IS estimator
# E_{x~p}[q(x)/p(x)] has variance equal to the chi-square divergence of q from p, which for a
# heavy-tailed target and a Gaussian q is astronomically large (measured: e^17 on the toy
# above), so it returns noise rather than an answer.  Quadrature is deterministic, and on a
# mildly non-Gaussian target a modest grid resolves it exactly.

MIX_WEIGHTS = np.array([0.55, 0.45])
MIX_MEANS = np.array([[0., 0., 0.], [1.2, -0.8, 0.4]])
MIX_STD = np.array([[1., 0.8, 1.1], [0.7, 1.3, 0.9]])


def mild_logpdf(x):
    """Exact log-density of a 2-component, correlated-free mixture: non-Gaussian, light tails."""
    x = np.asarray(x, dtype='f8')
    terms = []
    for weight, mean, std in zip(MIX_WEIGHTS, MIX_MEANS, MIX_STD):
        z = (x - mean) / std
        terms.append(np.log(weight) - 0.5 * z @ z - np.sum(np.log(std))
                     - 0.5 * len(x) * np.log(2 * np.pi))
    a, b = terms
    return float(np.logaddexp(a, b))


def mild_samples(nsamples, ndim, seed):
    rng = np.random.default_rng(seed)
    component = rng.choice(2, nsamples, p=MIX_WEIGHTS)
    draw = rng.normal(size=(nsamples, ndim))
    return np.where(component[:, None] == 0,
                    MIX_MEANS[0][:ndim] + MIX_STD[0][:ndim] * draw,
                    MIX_MEANS[1][:ndim] + MIX_STD[1][:ndim] * draw)


@pytest.mark.parametrize('kind', BACKENDS)
@pytest.mark.parametrize('ndim,ngrid', [(2, 241), (3, 81)])
def test_integrates_to_one_by_quadrature(kind, ndim, ngrid):
    r"""The fitted density must integrate to 1: :math:`\int q(x)\,\mathrm{d}^d x = 1`.

    No support box, so nothing is truncated and the answer is exactly 1.  A dropped
    log-determinant -- the classic normalizing-flow bug, and the reason to prefer a library
    ``log_prob`` over ``bijection.transform`` plus a base log-pdf -- shows up here as a
    systematic offset and nowhere else so cheaply.
    """
    names = NAMES[:ndim]
    X = mild_samples(4000, ndim, 21)
    density = fit_density(X, names, kind=kind, val_prop=0.25, seed=3, verbose=False,
                          **({'ncomponents': 12} if kind == 'gmm' else
                             {'knots': 6, 'max_epochs': 60} if kind == 'flow' else {}))
    axes = [np.linspace(X[:, dim].min() - 3., X[:, dim].max() + 3., ngrid) for dim in range(ndim)]
    grids = np.meshgrid(*axes, indexing='ij')
    points = np.column_stack([grid.ravel() for grid in grids])
    cell = float(np.prod([axis[1] - axis[0] for axis in axes]))
    # in chunks: a single log_prob_batch over a 3-D grid needs a few hundred MB of
    # intermediates, which is exactly the allocation a shared CI runner cannot spare
    total = 0.
    for start in range(0, points.shape[0], 50000):
        block = jnp.asarray(points[start:start + 50000])
        total += float(np.sum(np.exp(jnp.clip(density.log_prob_batch(block), -700., 700.))))
    integral = total * cell
    assert integral == pytest.approx(1., abs=0.02), f'{kind} integrates to {integral:.4f} in {ndim}-D'


def test_support_box_truncates_and_nothing_else():
    """Inside the box a masked density must equal the unmasked one exactly; outside, -inf.

    Truncating to the first-stage prior box is the intended behaviour -- the chain cannot
    leave it, so any density fitted to the chain is unconstrained there, and letting it leak
    past the prior is exactly what the mean/covariance reduction gets wrong.  What must not
    happen is the masked density differing from the unmasked one *inside* the box.
    """
    toy = _module_toy()
    fit_options = dict(kind='gmm', val_prop=0.25, seed=3, verbose=False, ncomponents=12)
    masked = fit_density(toy['train'], NAMES, limits=toy['limits'], **fit_options)
    free = fit_density(toy['train'], NAMES, **fit_options)
    # Probe both branches deliberately.  A box padded to the sample range is wide enough that
    # fresh draws from the toy all land inside it, which would leave the -inf branch untested
    # and the assertion vacuous -- so half the probe is pushed past the boundary on purpose.
    draws = _toy_samples(1000, 77)
    pushed = np.where(draws[:500] > toy['train'].mean(axis=0),
                      toy['limits'][:, 1] + 1., toy['limits'][:, 0] - 1.)
    probe = np.concatenate([draws, pushed])
    ratio = (jnp.asarray(masked.log_prob_batch(jnp.asarray(probe)))
             - jnp.asarray(free.log_prob_batch(jnp.asarray(probe))))
    inside = np.all((probe >= toy['limits'][:, 0]) & (probe <= toy['limits'][:, 1]), axis=1)
    assert 0. < inside.mean() < 1., f'the probe exercised one branch only ({inside.mean():.0%} inside)'
    np.testing.assert_allclose(np.asarray(ratio)[inside], 0., atol=1e-12,
                               err_msg='the mask changed the density inside the box')
    assert np.all(np.asarray(ratio)[~inside] == -np.inf), 'density leaked past the support box'


@pytest.mark.parametrize('kind', BACKENDS)
def test_gradient_matches_finite_differences(kind, fitted):
    """AD through the density must equal a central difference: catches a missing log-det."""
    density = _density(kind, fitted)
    path = lambda theta: jnp.array([1. + theta, 1. - 0.3 * theta, 1. + 0.5 * theta,
                                    0.01 * theta, 1. + 0.2 * theta])
    theta = jnp.array(0.004)
    automatic = float(jax.grad(lambda t: density.log_prob(path(t)))(theta))
    step = 1e-7
    numeric = float((density.log_prob(path(theta + step)) - density.log_prob(path(theta - step))) / (2 * step))
    assert automatic == pytest.approx(numeric, rel=1e-5), (
        f'{kind}: AD {automatic} vs finite differences {numeric} -- a log-Jacobian is missing')


@pytest.mark.parametrize('kind', BACKENDS)
def test_vmap_and_jit_of_grad(kind, fitted, toy):
    density = _density(kind, fitted)
    batch = jnp.asarray(toy['test'][:16])
    assert jax.vmap(density.log_prob)(batch).shape == (batch.shape[0],)
    assert np.all(np.isfinite(jax.jit(jax.grad(lambda x: density.log_prob(x)))(batch[0])))


@pytest.mark.parametrize('kind', BACKENDS)
def test_moments_are_not_off_by_a_factor_of_ncomponents(kind, fitted, toy):
    """``covariance()`` must be the same order as the sample covariance of what was fitted.

    This is the test that catches a mixture whose law-of-total-covariance forgot the mixing
    weight on the within-component term: the density stays perfectly good, so nothing else
    notices, but the reported covariance comes out roughly ``ncomponents`` times too large --
    and that number feeds proposals, plots and any downstream covariance-based diagnostic.
    The toy's own tails are heavy enough that only the order of magnitude is checkable here;
    ``test_mixture_moments_on_a_gaussian_chain`` pins it tightly.
    """
    density = _density(kind, fitted)
    empirical = np.cov(toy['train'].T)
    ratio = np.diag(density.covariance()) / np.diag(empirical)
    assert np.all(ratio < 2.), f'{kind}: variance ratio {ratio} -- the mixture weights look wrong'
    np.testing.assert_allclose(density.mean(), toy['train'].mean(axis=0), rtol=0.1, atol=1e-3)


def test_mixture_moments_on_a_gaussian_chain():
    """On a Gaussian chain the mixture's moments must match the sample moments tightly."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(3000, NDIM)) * SCALE + LOC
    density = fit_density(X, NAMES, kind='gmm', val_prop=0.25, seed=1, verbose=False, ncomponents=8)
    sample_covariance = np.cov(X.T)
    # An absolute tolerance is required: the toy's dimensions are independent, so the
    # off-diagonal entries are zero to within sampling noise and a relative one is not
    # meaningful there.
    np.testing.assert_allclose(density.covariance(), sample_covariance, rtol=0.1,
                               atol=0.05 * np.abs(np.diag(sample_covariance)).max())
    np.testing.assert_allclose(density.mean(), X.mean(axis=0), rtol=0.02, atol=1e-4)


@pytest.mark.parametrize('kind', BACKENDS)
def test_support_is_respected(kind, fitted, toy):
    """Outside the box the density must not invent mass -- a chain cannot leave its prior."""
    density = _density(kind, fitted)
    outside = np.where(toy['train'][0] > toy['train'].mean(axis=0),
                       toy['limits'][:, 1] + 1., toy['limits'][:, 0] - 1.)
    assert float(density.log_prob(jnp.asarray(outside))) == -np.inf
    inside = toy['limits'].mean(axis=1)
    assert np.isfinite(float(density.log_prob(jnp.asarray(inside))))
    # and the batched entry point must agree with the single-point one
    np.testing.assert_allclose(
        np.asarray(density.log_prob_batch(jnp.asarray(toy['test'][:8]))),
        [float(density.log_prob(jnp.asarray(x))) for x in toy['test'][:8]], atol=1e-12)


@pytest.mark.parametrize('kind', BACKENDS)
def test_log_prob_rejects_a_batch(kind, fitted, toy):
    """A batch handed to the single-point entry point must raise, not broadcast silently."""
    density = _density(kind, fitted)
    with pytest.raises(ValueError, match='single point'):
        density.log_prob(jnp.asarray(toy['test'][:4]))


@pytest.mark.parametrize('kind', BACKENDS)
def test_serialization_round_trip(kind, fitted, toy, tmp_path):
    """write -> read must be exact, and must survive float64."""
    density = _density(kind, fitted)
    filename = str(tmp_path / f'density_{kind}.h5')
    density.write(filename)
    assert os.path.getsize(filename) > 0
    reloaded = Density.read(filename)
    assert type(reloaded) is type(density)
    assert list(reloaded.names) == list(density.names)
    batch = jnp.asarray(toy['test'][:100])
    np.testing.assert_allclose(np.asarray(reloaded.log_prob_batch(batch)),
                               np.asarray(density.log_prob_batch(batch)), atol=1e-12, rtol=0)
    assert reloaded.log_prob(batch[0]).dtype == jnp.float64


def test_cache_hit_and_miss(toy, tmp_path):
    """The cache must key on the data *and* every hyper-parameter, not just the backend name."""
    filename = str(tmp_path / 'cached.h5')
    options = dict(kind='gmm', limits=toy['limits'], val_prop=0.25, verbose=False)
    first = fit_density(toy['train'], NAMES, filename=filename, seed=3,
                        select_from=[{'ncomponents': 12}], **options)
    first.attrs['marker'] = 'first'
    first.write(filename)
    again = fit_density(toy['train'], NAMES, filename=filename, seed=3,
                        select_from=[{'ncomponents': 12}], **options)
    assert again.attrs.get('marker') == 'first', 'an identical request must be served from the cache'
    changed = fit_density(toy['train'], NAMES, filename=filename, seed=3,
                          select_from=[{'ncomponents': 8}], **options)
    assert changed.attrs.get('marker') is None, 'a different capacity must refit, not reuse the cache'
    other = _toy_samples(N_TRAIN, 99)
    other_pad = 0.25 * np.ptp(other, axis=0)
    other_limits = np.column_stack([other.min(axis=0) - other_pad, other.max(axis=0) + other_pad])
    moved = fit_density(other, NAMES, filename=filename, seed=3, limits=other_limits,
                        val_prop=0.25, verbose=False, select_from=[{'ncomponents': 12}])
    assert moved.attrs.get('marker') is None, 'different data must refit, not reuse the cache'


def test_support_box_that_does_not_bracket_the_data_raises(toy):
    """A too-tight box would mask everything to -inf and leave model selection with no winner."""
    tight = np.column_stack([toy['train'].mean(axis=0) - 1e-9, toy['train'].mean(axis=0) + 1e-9])
    with pytest.raises(ValueError, match='does not contain the samples'):
        fit_density(toy['train'], NAMES, kind='gmm', limits=tight, verbose=False)


# ── layer 2: is a non-Gaussian density actually worth it here? ───────────────

def test_gaussian_baseline_loses_to_truth(toy, fitted):
    """The premise of the whole exercise, measured: mean/cov is measurably worse than truth."""
    gaussian = fitted['gaussian']
    excess = gaussian.heldout_nll(toy['test']) - toy['true_nll']
    assert excess > 0.5, f'the toy is not non-Gaussian enough to test anything (excess {excess:.3f})'


@pytest.mark.parametrize('kind', [k for k in BACKENDS if k != 'gaussian'])
def test_backend_recovers_what_the_gaussian_loses(kind, fitted, toy):
    """A flexible density must close most of the gap, on data it was not fitted to."""
    density = _density(kind, fitted)
    gaussian_excess = fitted['gaussian'].heldout_nll(toy['test']) - toy['true_nll']
    excess = density.heldout_nll(toy['test']) - toy['true_nll']
    assert excess < 0.5 * gaussian_excess, (
        f'{kind} recovers too little: {excess:.3f} nats over truth vs {gaussian_excess:.3f} for the Gaussian')


def test_no_spurious_structure_on_a_gaussian_chain():
    """On a genuinely Gaussian chain the flexible backend must not claim a gain.

    The overfitting guard: a density that always "wins" is fitting noise, and would quietly
    distort a real analysis whose compressed posterior happens to be Gaussian (BAO, say).
    """
    rng = np.random.default_rng(5)
    X = rng.normal(size=(2000, NDIM)) * SCALE + LOC
    holdout = rng.normal(size=(800, NDIM)) * SCALE + LOC
    gaussian = fit_density(X, NAMES, kind='gaussian', val_prop=0.25, verbose=False)
    mixture = fit_density(X, NAMES, kind='gmm', val_prop=0.25, seed=0, verbose=False,
                          select_from=[{'ncomponents': k} for k in (1, 4)])
    assert abs(gaussian.heldout_nll(holdout) - mixture.heldout_nll(holdout)) < 0.1


def test_credible_region_coverage(kind='gmm', fitted=None, toy=None):
    """Highest-density regions must contain the fraction of held-out samples they claim.

    The calibration test that matters for a likelihood: if the 68% region holds 80% of the
    mass, the density is too narrow and the second-stage posterior will be over-confident.
    """
    toy = toy or _module_toy()
    density = fit_density(toy['train'], NAMES, kind='gmm', limits=toy['limits'],
                          val_prop=0.25, seed=3, verbose=False, ncomponents=12)
    logpdf_test = np.asarray(density.log_prob_batch(jnp.asarray(toy['test'])))
    logpdf_train = np.asarray(density.log_prob_batch(jnp.asarray(toy['train'])))
    for level in (0.68, 0.95):
        threshold = np.quantile(logpdf_train, 1. - level)
        contained = float(np.mean(logpdf_test >= threshold))
        assert abs(contained - level) < 0.04, f'the {level:.0%} region holds {contained:.1%} of held-out samples'


def _module_toy():
    train, test = _toy_samples(N_TRAIN, 11), _toy_samples(N_TEST, 12)
    pad = 0.25 * np.ptp(train, axis=0)
    return dict(train=train, test=test, limits=np.column_stack([train.min(0) - pad, train.max(0) + pad]),
                true_nll=-float(np.mean([true_logpdf(x) for x in test])))


def test_limits_and_priors_are_read_off_the_chain(tmp_path):
    """A chain carries its first-stage priors through HDF5; a fit must use them unprompted.

    ``Parameter.__getstate__(to_file=True)`` serializes the prior into the file's JSON metadata,
    so a chain read back from disk is self-describing -- which is what makes
    ``fit_density(chain, names)`` with no ``limits`` and no ``priors`` the normal call rather
    than an under-specified one.  An unbounded prior must produce no box at all.
    """
    from desilike.parameter import Parameter
    from desilike.samples.chain import MCSamples
    rng = np.random.default_rng(4)
    boxes = {'a': (0.9, 1.1), 'b': (0.95, 1.05), 'c': (-0.2, 0.2)}
    names = ['a', 'b', 'c']
    centers = {'a': 1.0, 'b': 1.0, 'c': 0.0}      # inside each box, or the support guard fires
    X = np.column_stack([rng.normal(size=1200) * 0.01 + centers[name] for name in names])
    params = [Parameter(name, value=X[:, index], prior=dict(limits=boxes[name]))
              for index, name in enumerate(names)]
    chain = MCSamples(params)
    filename = str(tmp_path / 'chain_with_priors.h5')
    if hasattr(chain, 'write'):
        chain.write(filename)
        chain = MCSamples.read(filename)      # the round trip is the thing under test
    density = fit_density(chain, names, kind='gmm', val_prop=0.25, seed=0, verbose=False,
                          ncomponents=4)
    np.testing.assert_allclose(density.limits, np.array([boxes[name] for name in names]))
    recovered = density.priors()
    assert recovered is not None and recovered['a'].limits == boxes['a']
    # recorded on the density, so a likelihood built from the density alone still knows them
    density.write(str(tmp_path / 'density_with_priors.h5'))
    reloaded = Density.read(str(tmp_path / 'density_with_priors.h5'))
    assert reloaded.priors()['c'].limits == boxes['c']

    params[2] = Parameter('c', value=X[:, 2], prior=dict(dist='norm', loc=0., scale=1.))
    unbounded = fit_density(MCSamples(params), names, kind='gmm', val_prop=0.25, seed=0,
                            verbose=False, ncomponents=4)
    assert unbounded.limits is None, 'an unbounded prior must not invent a support box'


@pytest.mark.parametrize('kind', BACKENDS)
def test_coverage_reproduces_nominal_levels_on_its_own_draws(kind, fitted):
    """``coverage`` calibrated under q, measured on fresh draws from q, must return the levels.

    This is the implementation check for :meth:`~desilike.densities.base.Density.coverage`:
    with samples from exactly the density under test, every nominal region must hold its own
    mass.  It needs no truth and no split, so it runs for every backend.
    """
    density = _density(kind, fitted)
    own = np.asarray(density.sample(jax.random.key(99), (8000,)))
    row = density.coverage(own, ncalibrate=20000, seed=1)
    for level in (0.68, 0.95):
        assert abs(row[str(level)] - level) < 0.02, (
            f'{kind}: its own 100*{level:.0f}% region holds {row[str(level)]:.3f}')


def test_coverage_flags_an_over_dispersed_density():
    """On data from a known mild mixture, coverage is nominal; inflate the covariances and it
    must read high -- the model's regions have grown beyond where the data live.

    The converse (too narrow reads low) is deliberately *not* asserted: with heavy tails a
    shrunken fit can still be over-dispersed where it matters, and on the real ShapeFit chain
    the two coverage statistics disagreed in exactly this way.  The gap between this statistic
    and the train/test-quantile version is the diagnostic, not either number alone.
    """
    from desilike.densities.gmm import GaussianMixtureDensity
    rng = np.random.default_rng(21)
    X = mild_samples(4000, 3, 21)
    heldout = mild_samples(2000, 3, 22)
    density = fit_density(X, ['a', 'b', 'c'], kind='gmm', val_prop=0.25, seed=3, verbose=False,
                          ncomponents=8)
    row = density.coverage(heldout, ncalibrate=20000, seed=5)
    for level in (0.68, 0.95):
        assert abs(row[str(level)] - level) < 0.04, f'well-specified fit mis-calibrated: {row}'
    wide = GaussianMixtureDensity(weights=density.weights, means=density.means,
                                  covariances=np.asarray(density.covariances) * 4.,
                                  names=density.names, loc=density.loc, scale=density.scale,
                                  limits=density.limits)
    row_wide = wide.coverage(heldout, ncalibrate=20000, seed=5)
    assert row_wide['0.68'] > row['0.68'] + 0.05, (
        f'an over-dispersed density must read high: {row_wide["0.68"]:.3f} vs {row["0.68"]:.3f}')


def test_flow_kind_selects_that_architecture():
    """``kind='coupling'`` must fit a coupling flow, not silently fall back to the MAF default.

    The registry maps several ``kind`` values to the same backend class, so the name has to be
    forwarded as the architecture; without that, ``kind='bnaf'`` would produce a masked
    autoregressive flow whose cache then claims to be what was asked for.
    """
    pytest.importorskip('flowjax')
    rng = np.random.default_rng(6)
    X = mild_samples(1500, 3, 6)
    density = fit_density(X, ['a', 'b', 'c'], kind='coupling', val_prop=0.25, seed=1,
                          verbose=False, flow_layers=2, nn_width=8, max_epochs=5)
    assert density.config['kind'] == 'coupling'
    assert density.attrs['config'] and '"kind": "coupling"' in density.attrs['config']


def test_cache_key_includes_the_thinning(tmp_path):
    """Thinning changes the training data, so it must invalidate the cache."""
    toy = _module_toy()
    filename = str(tmp_path / 'thinned.h5')
    common = dict(kind='gmm', limits=toy['limits'], val_prop=0.25, verbose=False,
                  select_from=[{'ncomponents': 4}])
    first = fit_density(toy['train'], NAMES, thin=4, filename=filename, **common)
    first.attrs['marker'] = 'thin4'
    first.write(filename)
    same = fit_density(toy['train'], NAMES, thin=4, filename=filename, **common)
    assert same.attrs.get('marker') == 'thin4', 'identical thinning must hit the cache'
    other = fit_density(toy['train'], NAMES, thin=9, filename=filename, **common)
    assert other.attrs.get('marker') is None, 'a different thinning must refit, not reuse'
