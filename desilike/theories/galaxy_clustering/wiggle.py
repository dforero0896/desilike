"""Wiggle-depth (``dgam``) estimators for the ShapeFit compression: the final, minimal set.

This module is deliberately small.  It contains exactly the estimators that ship as
production code in :class:`~desilike.theories.galaxy_clustering.template.ShapeFitTheory`:
the RMS of the log wiggle depth ln W = ln P - ln P_now over a window fixed in *phase*
x = k r_d, optionally after removing a low-order polynomial in ln k.  Everything else that
was explored (matched-filter projections, orthogonalised/linearised-ML projections, joint
whole-basis projections, single-peak amplitudes, quadrature demodulation) lives outside the
package as *diagnostics* (``study/wiggle_diagnostics.py`` in the accompanying work), because
a compression parameter that needs a configuration to be defined is against the spirit of
ShapeFit; see the report ``shapefit_gamma.html``, Appendices A.4-A.7.

Each estimator is a pure function pair, so it can be tested, jit-compiled and compared
without a Boltzmann engine or a cosmology object:

    state = EST.prepare(log_k_fid, log_pk_fid, log_pknow_fid, rd_fid, **opts)  # numpy, once
    dgam  = EST.apply(state, log_k_fine, log_pk_fine, log_pknow_fine, rd)      # jax, per call

Conventions
-----------
* ``log_pk``, ``log_pknow`` are ln P(k) and ln P_now(k) on one static dense k grid
  (``default_fine_grid()``); ``rd`` is the sound horizon of the evaluated cosmology.
* The window is [kmin, kmax] in phase: k_i = x_i / rd with x_i fixed, so it follows the
  wiggle phase of each cosmology.  This removes the legacy fixed-k window's response to a
  pure change of r_d (d gamma / d ln s ~ -0.25 for a dilation whose true gamma is 0).
* Normalisation: dgam = ratio - 1, matching the template's ln W -> (1 + dgam) ln W.
  ('rms_legacy' keeps the old ln(ratio) convention to reproduce existing chains exactly.)

Accuracy (max |dgam - dgam_ML| over the nine-cosmology suite of the report, in units of the
reference mock's sigma(dgam) = 0.113):  'rms_legacy' 0.42, 'rms_phase' 0.39 (but only 0.07
for variations within current cosmological bounds), 'rms_phase_detrend' 0.04.
"""
import numpy as np
import jax.numpy as jnp

__all__ = ['WiggleEstimator', 'WIGGLE_ESTIMATORS', 'get_wiggle_estimator',
           'default_fine_grid', 'wiggle']


def default_fine_grid():
    """Static dense k grid [h/Mpc] the estimators interpolate from (600 log-spaced points)."""
    return np.geomspace(0.005, 0.6, 600)


def wiggle(log_pk, log_pknow):
    """ln W(k) = ln P(k) - ln P_now(k), the log wiggle depth."""
    return log_pk - log_pknow


def _interp(log_k_fine, y, k):
    return jnp.interp(jnp.log(k), log_k_fine, y)


def _detrend(log_k, y, deg):
    """Remove a polynomial of degree ``deg`` in ln k (jax-safe)."""
    if not deg:
        return y
    return y - jnp.polyval(jnp.polyfit(log_k, y, deg), log_k)


class WiggleEstimator:
    """A named wiggle-depth estimator: ``prepare`` (numpy, once) + ``apply`` (jax, per call).

    Parameters
    ----------
    name : str
        Registry key, as passed to ``ShapeFitTheory(..., gam_estimator=name)``.
    prepare : callable
        ``prepare(log_k_fid, log_pk_fid, log_pknow_fid, rd_fid, **opts) -> state``.
    apply : callable
        ``apply(state, log_k_fine, log_pk_fine, log_pknow_fine, rd) -> dgam``; pure jax.
    definition : str
        The estimator written out, for documentation.
    traits : dict
        ``phase_anchored``, ``smooth_robust``, ``linear_in_lnW``, ``max_err_vs_ML``, ``notes``.
    opts : dict
        Default keyword options forwarded to ``prepare``.
    """

    def __init__(self, name, prepare, apply, definition, traits, **opts):
        self.name = name
        self._prepare = prepare
        self._apply = apply
        self.definition = definition
        self.traits = traits
        self.opts = opts

    def prepare(self, log_k_fid, log_pk_fid, log_pknow_fid, rd_fid, **opts):
        import inspect
        o = dict(self.opts)
        o.update(opts)
        par = inspect.signature(self._prepare).parameters
        if not any(q.kind is q.VAR_KEYWORD for q in par.values()):
            o = {k: v for k, v in o.items() if k in par}   # ignore opts this estimator lacks
        state = self._prepare(log_k_fid, log_pk_fid, log_pknow_fid, rd_fid, **o)
        state['name'] = self.name
        return state

    def apply(self, state, log_k_fine, log_pk_fine, log_pknow_fine, rd):
        return self._apply(state, log_k_fine, log_pk_fine, log_pknow_fine, rd)

    def __repr__(self):
        return 'WiggleEstimator(%r)' % self.name


def _prepare_rms(log_k_fid, log_pk_fid, log_pknow_fid, rd_fid, kmin=0.04, kmax=0.35, n=200,
                 phase=True, detrend=0, log_ratio=False):
    x = np.geomspace(kmin * rd_fid, kmax * rd_fid, n)
    k0 = x / rd_fid
    lw0 = np.interp(np.log(k0), log_k_fid, log_pk_fid - log_pknow_fid)
    if detrend:
        lw0 = lw0 - np.polyval(np.polyfit(np.log(k0), lw0, detrend), np.log(k0))
    return dict(x=x, k0=k0, phase=bool(phase), detrend=int(detrend), log_ratio=bool(log_ratio),
                norm=float(np.sqrt(np.mean(lw0 ** 2))))


def _apply_rms(state, log_k_fine, log_pk_fine, log_pknow_fine, rd):
    # phase-anchored: the window follows the cosmology's wiggle phase, k = x / rd.
    # fixed-k (legacy): the window is in k and must NOT move with rd -- that movement is
    # precisely the r_d beat the anchored version removes.
    k = state['x'] / rd if state['phase'] else state['k0']
    lw = _detrend(jnp.log(k), _interp(log_k_fine, log_pk_fine - log_pknow_fine, k), state['detrend'])
    ratio = jnp.sqrt(jnp.mean(lw ** 2)) / state['norm']
    return jnp.log(ratio) if state['log_ratio'] else ratio - 1.


WIGGLE_ESTIMATORS = {}


def _register(est):
    WIGGLE_ESTIMATORS[est.name] = est
    return est


_register(WiggleEstimator(
    'rms_legacy', _prepare_rms, _apply_rms,
    'dgam = ln[ rms_{k in [0.04,0.35]}(ln W) / rms_fid ], 48-point fixed-k grid: reproduces the '
    'original shapefit-prime code bit-for-bit (fixed window, ln-ratio convention)',
    dict(phase_anchored=False, smooth_robust=False, linear_in_lnW=False, max_err_vs_ML=0.42,
         notes='kept only to reproduce existing chains; leaks d gamma/d ln s ~ -0.25 and is '
               'quadratic in smooth P/P_now mismatch (Sigma m_nu + 0.2 eV -> +0.045, truth +0.003)'),
    kmin=0.04, kmax=0.35, n=48, phase=False, log_ratio=True))

_register(WiggleEstimator(
    'rms_phase', _prepare_rms, _apply_rms,
    'dgam = rms_{x in [3.96,34.7]}(ln W) / rms_fid - 1 with x = k r_d, i.e. the window follows '
    'the wiggle phase of each cosmology',
    dict(phase_anchored=True, smooth_robust=False, linear_in_lnW=False, max_err_vs_ML=0.39,
         notes='the minimal fix: <= 0.07 sigma for variations within current bounds; still a '
               'norm, so a smooth P/P_now mismatch inflates it (0.39 sigma at Sigma m_nu + 0.2 eV)'),
    kmin=0.04, kmax=0.35, n=200, phase=True))

_register(WiggleEstimator(
    'rms_phase_detrend', _prepare_rms, _apply_rms,
    'dgam = rms_x(ln W - poly_3[ln W]) / rms_x(ln W_fid - poly_3[ln W_fid]) - 1, x = k r_d: '
    'the phase-anchored RMS after removing a cubic polynomial in ln k',
    dict(phase_anchored=True, smooth_robust=True, linear_in_lnW=False, max_err_vs_ML=0.04,
         notes='PACKAGE DEFAULT. Still just a norm: no basis, no linear algebra, no '
               'configuration; the detrend removes smooth filter mismatch (Sigma m_nu + 0.2 eV '
               '-> +0.003, truth +0.003). Like any norm it is blind to the wiggle sign.'),
    kmin=0.04, kmax=0.35, n=200, phase=True, detrend=3))


def get_wiggle_estimator(name):
    """Return the registered :class:`WiggleEstimator` called ``name``."""
    try:
        return WIGGLE_ESTIMATORS[str(name)]
    except KeyError:
        raise ValueError('unknown wiggle estimator %r; available: %s'
                         % (name, ', '.join(sorted(WIGGLE_ESTIMATORS)))) from None
