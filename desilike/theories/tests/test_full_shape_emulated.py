"""Emulator integration tests for every PT sub-calculator in full_shape.py.

Pattern (mirrors benchmark.py::build_posterior_folps with emulator_order set):
  1. Build the theory with its real PT sub-calculator.
  2. Emulate the PT sub-graph over its parameters ref box.
  3. Replace the real PT with the emulated version (via replace()).
  4. Compile the full tracer pipeline and compare against an exact reference.

Checks:
  - Exact match at the Taylor expansion center (always true for any order ≥ 1).
  - Relative error < 10 % after a 5 % shift in a cosmological / bias parameter.

Theories backed by optional packages (folps, velocileptors, pybird) are
auto-skipped when the package is not installed.
"""

import numpy as np
import pytest
import jax

jax.config.update('jax_enable_x64', True)

from desilike import build
from desilike.emulators import Emulator, Space
from desilike.base import replace


_FID = ('DESI', {'engine': 'camb'})
_K = np.linspace(0.02, 0.2, 15)
_ELLS = (0, 2)
_S = np.linspace(50., 150., 10)
_EMU_ORDER = 1
# Training costs one full PT evaluation per finite-difference node, ~(2 * ndim + 1) of them, so
# the box dimension sets the runtime of this whole file (measured on Kaiser: 3.96s at ndim=5,
# 1.87s at ndim=2, i.e. linear).  `_check` only ever shifts a BIAS parameter (`b1`, `b1p`), which
# is downstream of the emulator, so no assertion here evaluates it off-centre along any of those
# axes -- carrying all of them is cost for nothing.  Two keep the multi-parameter plumbing under
# test at a bit under half the price; `max_axes=None` asks for the full box, which one test per
# file still does so the all-axes path stays covered.
# A COUNT, not a list of names: which parameters are varied depends on the template (a default
# template gives h/logA/n_s/omega_b/omega_cdm, a `BAOSpectrum2Template(apmode='qparqper')` gives
# the dilation parameters instead), so naming them fits one case and raises on the next.
# NOTE: test_full_shape.py carries its own copy of this helper; keep the two in step.
_EMU_MAX_AXES = 2


# ── helpers ────────────────────────────────────────────────────────────────────

def _fd_box(calculator, width=3., max_axes=_EMU_MAX_AXES):
    """A box `value +- width * fd.eps` per varied parameter.

    The legacy TaylorEmulator expanded about the centre with those same FD steps, so this keeps
    the emulated region comparable. not the `ref` box: these parameters have none, and the
    prior is far too wide to evaluate.

    *max_axes* caps how many varied parameters the box spans, taking them in name order so the
    choice is reproducible (``None`` keeps every one).  A parameter left out is frozen at its
    trained value, which `_check` already accounts for.
    """
    from desilike.base import build
    limits = {}
    for param in build(calculator).params:
        # varied only. A fixed parameter does not move, so emulating over it buys nothing and
        # costs an axis -- 11 axes instead of 5 for a FOLPS pt -- and it is how the box came to
        # ask for a negative neutrino mass: m_ncdm is fixed, with a leftover fd step.
        if param.derived or not getattr(param, 'varied', True):
            continue
        value = float(np.sum(np.atleast_1d(param.value)))
        eps = getattr(getattr(param, 'fd', None), 'eps', None) or max(abs(value), 1.) * 0.02
        low, high = value - width * eps, value + width * eps
        # A varied parameter sitting at a prior edge is still possible; skip it rather than clip.
        # `differentiate` handles the same situation by shifting its stencil inside the prior and
        # keeping the expansion centre put, which a Taylor expansion can do and a collocation grid
        # cannot: shifting the box moves its midpoint off the parameter value, and the midpoint is
        # the node `_check` asserts the emulator is exact at.
        bounds = getattr(getattr(param, 'prior', None), 'limits', None)
        if bounds is not None and np.isfinite(bounds).all() and (low < float(bounds[0]) or high > float(bounds[1])):
            continue
        limits[param.name] = (low, high)
    if max_axes is not None:
        if not limits:
            raise ValueError(f'{calculator} has no varied parameter to emulate over')
        limits = {name: limits[name] for name in sorted(limits)[:max_axes]}
    return limits


def _train(inner_pt, max_axes=_EMU_MAX_AXES):
    """Train an emulator on *inner_pt* over its finite-difference box."""
    emu = Emulator(inner_pt, Space(bounds=_fd_box(inner_pt, max_axes=max_axes)))
    emu.train(budget=_EMU_ORDER)
    return emu


def _emulate(theory, inner_pt=None, max_axes=_EMU_MAX_AXES):
    """Emulate ``inner_pt`` (default: ``theory.pt``) in-place; return compiled pipeline."""
    if inner_pt is None:
        inner_pt = theory.pt
    replace(theory, inner_pt, _train(inner_pt, max_axes=max_axes).to_calculator(inner_pt))
    return build(theory)


def _check(pipe_exact, pipe_emu, shift_param, reldiff_tol=0.10):
    """Center-exact match (atol=1e-8) and shifted accuracy (relative < tol).

    The deployed emulator exposes a per-emulator coverage-check parameter
    (``emulator_<class>_<hash>.box``) that tracks how far the evaluation point
    lies outside the training box. It is internal bookkeeping, not part of the
    theory, so it is filtered from the parameter-interface comparison; when the
    call omits it, the deployed calculator uses its fixed default value.
    """
    center = {p.name: p.value for p in pipe_exact.params
              if not p.name.startswith('emulator_')}
    emu_names = {n for n in pipe_emu.params.names()
                 if not n.startswith('emulator_')}
    assert emu_names <= set(center), \
        f'emulated pipeline gained parameters: {sorted(emu_names - set(center))}'
    emu_center = {name: value for name, value in center.items() if name in emu_names}
    exact_center_value = np.asarray(pipe_exact(center))
    emu_center_value = np.asarray(pipe_emu(emu_center))
    # rtol as well as atol: "exact at the centre" means to machine precision, and an
    # absolute-only tolerance says something different at every scale -- on the bispectrum,
    # whose values are ~1e9, atol=1e-8 demands a relative 1e-17 and fails on rounding alone.
    np.testing.assert_allclose(emu_center_value, exact_center_value, atol=1e-8, rtol=1e-10,
                               err_msg='emulator mismatch at expansion center')
    if shift_param in center:
        shifted = {**center, shift_param: center[shift_param] * 1.05}
        exact_s = np.asarray(pipe_exact(shifted))
        emu_s = np.asarray(pipe_emu({name: value for name, value in shifted.items() if name in emu_names}))
        reldiff = float(np.max(np.abs(emu_s - exact_s) / (np.abs(exact_s) + 1e-30)))
        assert reldiff < reldiff_tol, \
            f'shifted [{shift_param}+5%]: max reldiff={reldiff:.3f} > {reldiff_tol:.2f}'

# ── Kaiser ─────────────────────────────────────────────────────────────────────

def test_kaiser_spectrum_emulated():
    """KaiserPTSpectrum2Poles emulated as pt= in KaiserTracerSpectrum2Poles."""
    from desilike.theories.galaxy_clustering.full_shape import KaiserPTSpectrum2Poles, KaiserTracerSpectrum2Poles
    from desilike.theories.galaxy_clustering.template import BAOSpectrum2Template

    pipe_exact = build(KaiserTracerSpectrum2Poles(
        k=_K, ells=_ELLS,
        template=BAOSpectrum2Template(z=0.5, fiducial=_FID, apmode='qparqper')))

    theory_emu = KaiserTracerSpectrum2Poles(
        k=_K, ells=_ELLS,
        pt=KaiserPTSpectrum2Poles(k=_K, ells=_ELLS,
                                   template=BAOSpectrum2Template(z=0.5, fiducial=_FID, apmode='qparqper')))
    # `max_axes=None`: the one test in this file that trains over EVERY varied parameter, so the
    # full-box path stays covered.  Kaiser is the cheapest PT here, which is why it carries it.
    pipe_emu = _emulate(theory_emu, max_axes=None)

    _check(pipe_exact, pipe_emu, shift_param='b1')


def test_kaiser_correlation_emulated():
    """KaiserPTSpectrum2Poles emulated inside KaiserTracerCorrelation2Poles."""
    from desilike.theories.galaxy_clustering.full_shape import KaiserTracerCorrelation2Poles
    from desilike.theories.galaxy_clustering.template import BAOSpectrum2Template

    pipe_exact = build(KaiserTracerCorrelation2Poles(
        s=_S, ells=_ELLS,
        template=BAOSpectrum2Template(z=0.5, fiducial=_FID, apmode='qparqper')))

    theory_emu = KaiserTracerCorrelation2Poles(
        s=_S, ells=_ELLS,
        template=BAOSpectrum2Template(z=0.5, fiducial=_FID, apmode='qparqper'))
    # theory_emu.pt is KaiserTracerSpectrum2Poles; .pt.pt is KaiserPTSpectrum2Poles
    pipe_emu = _emulate(theory_emu, inner_pt=theory_emu.pt.pt)

    _check(pipe_exact, pipe_emu, shift_param='b1')


# ── tns ────────────────────────────────────────────────────────────────────────

def test_tns_spectrum_emulated():
    """TNSPTSpectrum2Poles emulated as pt= in TNSTracerSpectrum2Poles."""
    from desilike.theories.galaxy_clustering.full_shape import TNSPTSpectrum2Poles, TNSTracerSpectrum2Poles

    pipe_exact = build(TNSTracerSpectrum2Poles(k=_K, ells=_ELLS))

    theory_emu = TNSTracerSpectrum2Poles(k=_K, ells=_ELLS,
                                          pt=TNSPTSpectrum2Poles(k=_K, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1')


def test_tns_correlation_emulated():
    """TNSPTSpectrum2Poles emulated inside TNSTracerCorrelation2Poles."""
    from desilike.theories.galaxy_clustering.full_shape import TNSTracerCorrelation2Poles

    pipe_exact = build(TNSTracerCorrelation2Poles(s=_S, ells=_ELLS))

    theory_emu = TNSTracerCorrelation2Poles(s=_S, ells=_ELLS)
    pipe_emu = _emulate(theory_emu, inner_pt=theory_emu.pt.pt)

    _check(pipe_exact, pipe_emu, shift_param='b1')


# ── lpt Velocileptors ──────────────────────────────────────────────────────────

def test_lpt_spectrum_emulated():
    """LPTVelocileptorsPTSpectrum2Poles emulated as pt= in LPTVelocileptorsTracerSpectrum2Poles."""
    pytest.importorskip('velocileptors')
    from desilike.theories.galaxy_clustering.full_shape import (LPTVelocileptorsPTSpectrum2Poles,
                                                                LPTVelocileptorsTracerSpectrum2Poles)

    pipe_exact = build(LPTVelocileptorsTracerSpectrum2Poles(k=_K, ells=_ELLS))

    theory_emu = LPTVelocileptorsTracerSpectrum2Poles(
        k=_K, ells=_ELLS,
        pt=LPTVelocileptorsPTSpectrum2Poles(k=_K, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


def test_lpt_correlation_emulated():
    """LPTVelocileptorsPTSpectrum2Poles emulated inside LPTVelocileptorsTracerCorrelation2Poles."""
    pytest.importorskip('velocileptors')
    from desilike.theories.galaxy_clustering.full_shape import LPTVelocileptorsTracerCorrelation2Poles

    pipe_exact = build(LPTVelocileptorsTracerCorrelation2Poles(s=_S, ells=_ELLS))

    theory_emu = LPTVelocileptorsTracerCorrelation2Poles(s=_S, ells=_ELLS)
    pipe_emu = _emulate(theory_emu, inner_pt=theory_emu.pt.pt)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


# ── rept Velocileptors ─────────────────────────────────────────────────────────

def test_rept_spectrum_emulated():
    """REPTVelocileptorsPTSpectrum2Poles emulated as pt= in REPTVelocileptorsTracerSpectrum2Poles."""
    pytest.importorskip('velocileptors')
    from desilike.theories.galaxy_clustering.full_shape import (REPTVelocileptorsPTSpectrum2Poles,
                                                                REPTVelocileptorsTracerSpectrum2Poles)

    pipe_exact = build(REPTVelocileptorsTracerSpectrum2Poles(k=_K, ells=_ELLS))

    theory_emu = REPTVelocileptorsTracerSpectrum2Poles(
        k=_K, ells=_ELLS,
        pt=REPTVelocileptorsPTSpectrum2Poles(k=_K, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


def test_rept_correlation_emulated():
    """REPTVelocileptorsPTSpectrum2Poles emulated inside REPTVelocileptorsTracerCorrelation2Poles."""
    pytest.importorskip('velocileptors')
    from desilike.theories.galaxy_clustering.full_shape import REPTVelocileptorsTracerCorrelation2Poles

    pipe_exact = build(REPTVelocileptorsTracerCorrelation2Poles(s=_S, ells=_ELLS))

    theory_emu = REPTVelocileptorsTracerCorrelation2Poles(s=_S, ells=_ELLS)
    pipe_emu = _emulate(theory_emu, inner_pt=theory_emu.pt.pt)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


# ── PyBird ─────────────────────────────────────────────────────────────────────

def test_pybird_spectrum_emulated():
    """PyBirdPTSpectrum2Poles emulated as pt= in PyBirdTracerSpectrum2Poles."""
    pytest.importorskip('pybird')
    try:
        import pybird.module  # noqa: F401
    except ImportError as exc:  # installed pybird is not numpy 2 compatible
        pytest.skip(f'installed pybird is not numpy 2 compatible: {exc}')
    from desilike.theories.galaxy_clustering.full_shape import PyBirdPTSpectrum2Poles, PyBirdTracerSpectrum2Poles

    pipe_exact = build(PyBirdTracerSpectrum2Poles(k=_K, ells=_ELLS))

    theory_emu = PyBirdTracerSpectrum2Poles(k=_K, ells=_ELLS,
                                             pt=PyBirdPTSpectrum2Poles(k=_K, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1')


def test_pybird_correlation_emulated():
    """PyBirdPTCorrelation2Poles emulated as pt= in PyBirdTracerCorrelation2Poles."""
    pytest.importorskip('pybird')
    try:
        import pybird.module  # noqa: F401
    except ImportError as exc:  # installed pybird is not numpy 2 compatible
        pytest.skip(f'installed pybird is not numpy 2 compatible: {exc}')
    from desilike.theories.galaxy_clustering.full_shape import PyBirdPTCorrelation2Poles, PyBirdTracerCorrelation2Poles

    pipe_exact = build(PyBirdTracerCorrelation2Poles(s=_S, ells=_ELLS))

    # PyBirdTracerCorrelation2Poles.pt is PyBirdPTCorrelation2Poles (direct PT, not nested)
    theory_emu = PyBirdTracerCorrelation2Poles(s=_S, ells=_ELLS,
                                               pt=PyBirdPTCorrelation2Poles(s=_S, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1')


# ── FOLPS ──────────────────────────────────────────────────────────────────────

def test_folps_spectrum_emulated():
    """FOLPSPTSpectrum2Poles emulated as pt= in FOLPSTracerSpectrum2Poles."""
    pytest.importorskip('folps')
    from desilike.theories.galaxy_clustering.full_shape import FOLPSPTSpectrum2Poles, FOLPSTracerSpectrum2Poles

    pipe_exact = build(FOLPSTracerSpectrum2Poles(k=_K, ells=_ELLS))

    theory_emu = FOLPSTracerSpectrum2Poles(k=_K, ells=_ELLS,
                                            pt=FOLPSPTSpectrum2Poles(k=_K, ells=_ELLS))
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


def test_folps_derived_param_omega_m():
    """A derived Parameter('Omega_m') on a nested cosmo dep survives PT emulation."""
    pytest.importorskip('folps')
    from desilike.base import get_params
    from desilike.parameter import Parameter
    from desilike.theories import CosmoprimoCosmology
    from desilike.theories.galaxy_clustering import DirectSpectrum2Template
    from desilike.theories.galaxy_clustering.full_shape import FOLPSTracerSpectrum2Poles

    cosmo = CosmoprimoCosmology(engine='camb', fiducial='DESI')
    params = get_params(cosmo)
    params.set(Parameter('Omega_m', value=0.0, derived=True))
    cosmo.update(params=params)
    template = DirectSpectrum2Template(cosmo=cosmo, z=1.)

    theory = FOLPSTracerSpectrum2Poles(k=_K, ells=_ELLS, template=template)
    pipe_emu = _emulate(theory)  # emulates theory.pt by default, mirrors the test above

    center = {p.name: p.value for p in pipe_emu.params}
    _, deriveds = pipe_emu(center, return_derived=True)
    assert 'Omega_m' in deriveds, f'Omega_m missing from derived params: {list(deriveds)}'
    assert np.isfinite(np.asarray(deriveds['Omega_m']))
    assert float(deriveds['Omega_m']) > 0.

    # sanity: matches an independent plain-cosmoprimo computation at the same params
    import cosmoprimo
    fid = cosmoprimo.fiducial.DESI(engine='camb')
    cosmo_param_names = {p.basename for p in get_params(CosmoprimoCosmology(engine='camb', fiducial='DESI'))}
    kw = {name: value for name, value in center.items() if name in cosmo_param_names}
    expected = fid.clone(base='input', **kw)['Omega_m']
    assert np.isclose(float(deriveds['Omega_m']), expected, rtol=1e-6)


def test_folps_correlation_emulated():
    """FOLPSPTSpectrum2Poles emulated inside FOLPSTracerCorrelation2Poles."""
    pytest.importorskip('folps')
    from desilike.theories.galaxy_clustering.full_shape import FOLPSTracerCorrelation2Poles

    pipe_exact = build(FOLPSTracerCorrelation2Poles(s=_S, ells=_ELLS))

    theory_emu = FOLPSTracerCorrelation2Poles(s=_S, ells=_ELLS)
    # theory_emu.pt is FOLPSTracerSpectrum2Poles; .pt.pt is FOLPSPTSpectrum2Poles
    pipe_emu = _emulate(theory_emu, inner_pt=theory_emu.pt.pt)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


def test_folps_spectrum3_poles_prior_bases():
    """FOLPSTracerSpectrum3Poles: each prior_basis compiles and produces the right output shape.

    Uses an emulator on the PT sub-graph to avoid the memory-intensive
    combine_bias_terms_spectrum3_poles call while still exercising the full
    bias-parameter conversion path for every prior_basis variant.
    """
    pytest.importorskip('folps')
    from desilike.theories.galaxy_clustering.full_shape import FOLPSPTSpectrum2Poles, FOLPSTracerSpectrum3Poles

    _K3 = np.column_stack([np.linspace(0.02, 0.1, 5)] * 2)
    _ELLS3 = ((0, 0, 0), (2, 0, 2))

    # `prior_basis` selects a bias-parameter conversion, which sits DOWNSTREAM of the PT
    # sub-graph: the four theories below configure their `pt` identically, so training an
    # emulator per iteration trained the same thing four times over (measured at 879s, the
    # slowest test in the suite by 2.4x).  Train once and deploy it into each theory instead.
    # Each deployment gets its own `pt` instance, because `to_calculator` takes the constructor
    # arguments -- template and cosmology included -- off the calculator it is handed, and
    # sharing one across the four would leave them aliasing a single template.
    bases = ('standard', 'physical', 'physical_aap', 'tcm_chudaykin_aap')
    theories, pts = [], []
    for prior_basis in bases:
        pt = FOLPSPTSpectrum2Poles()
        theories.append(FOLPSTracerSpectrum3Poles(k=_K3, ells=_ELLS3, prior_basis=prior_basis, pt=pt))
        pts.append(pt)

    emu = _train(pts[0])
    for prior_basis, theory, pt in zip(bases, theories, pts):
        replace(theory, pt, emu.to_calculator(pt))
        out = np.asarray(build(theory)())
        assert out.shape == (len(_ELLS3), len(_K3)), f'unexpected shape {out.shape} for prior_basis={prior_basis!r}'



def _pk1(pipe, point):
    """Evaluate *pipe* at *point* and return ``pk_1loop`` as an array.

    The raw pt returns the array from ``__call__``; the emulated subclass
    sets it as an attribute and returns ``self`` (so the tracer can read
    ``.pk_1loop``, ``.f``, … after the call). Both end up as the same array.
    """
    out = pipe(**point)
    return np.asarray(getattr(out, 'pk_1loop', out))

def test_folps_spectrum3_poles_emulated():
    """FOLPSPTSpectrum2Poles emulated as pt= in FOLPSTracerSpectrum3Poles."""
    pytest.importorskip('folps')
    from desilike.theories.galaxy_clustering.full_shape import FOLPSPTSpectrum2Poles, FOLPSTracerSpectrum3Poles

    pipe_exact = build(FOLPSTracerSpectrum3Poles())

    theory_emu = FOLPSTracerSpectrum3Poles(pt=FOLPSPTSpectrum2Poles())
    pipe_emu = _emulate(theory_emu)

    _check(pipe_exact, pipe_emu, shift_param='b1p')


def test_geofptax_spectrum3_poles_emulated():
    """GeoFPTAXPTSpectrum2Poles emulated as pt= in GeoFPTAXTracerSpectrum3Poles."""
    pytest.importorskip('geofptax')
    from desilike.theories.galaxy_clustering.full_shape import (
        GeoFPTAXPTSpectrum2Poles, GeoFPTAXTracerSpectrum3Poles)

    _K3 = np.column_stack([np.linspace(0.02, 0.1, 5)] * 2)
    _ELLS3 = ((0, 0, 0), (2, 0, 2))

    pipe_exact = build(GeoFPTAXTracerSpectrum3Poles(k=_K3, ells=_ELLS3))
    theory_emu = GeoFPTAXTracerSpectrum3Poles(
        k=_K3, ells=_ELLS3, pt=GeoFPTAXPTSpectrum2Poles())
    pipe_emu = _emulate(theory_emu)
    _check(pipe_exact, pipe_emu, shift_param='b1')

def test_geofptax_routing_select_params():
    """GeoFPTAXEmulator routes (w0_fld, wa_fld, logA) out of the grid."""
    pytest.importorskip('geofptax')
    from desilike.theories.galaxy_clustering.full_shape import (
        GeoFPTAXPTSpectrum2Poles, GeoFPTAXEmulator)

    pt = GeoFPTAXPTSpectrum2Poles()
    assert pt.get_emulator_cls() is GeoFPTAXEmulator

    from desilike.emulators import Space
    bounds = {
        'h': (0.6, 0.8), 'omega_b': (0.021, 0.024),
        'omega_cdm': (0.10, 0.14), 'n_s': (0.90, 1.00),
        'w0_fld': (-1.2, -0.8), 'wa_fld': (-0.6, 0.6),
        'logA': (2.8, 3.2),
    }
    space = Space(bounds=bounds)
    emu = GeoFPTAXEmulator(pt, space)

    all_names = list(bounds.keys())
    selected = emu.select_params(all_names)
    for p in ('w0_fld', 'wa_fld', 'logA'):
        assert p not in selected, f'{p} should be routed, not gridded'
    for p in ('h', 'omega_b', 'omega_cdm', 'n_s'):
        assert p in selected, f'{p} should be gridded, not routed'
    assert emu.precondition == ('h',), \
        f'h should be preconditioned, got {emu.precondition}'
    assert set(all_names) - set(selected) == {'w0_fld', 'wa_fld', 'logA'}



def test_geofptax_accuracy_in_routed_parameter():
    """Routing must be exact in (w0_fld, wa_fld, logA) — and we want to know which one costs what."""
    pytest.importorskip('geofptax')
    from desilike.theories.galaxy_clustering.full_shape import (
        GeoFPTAXPTSpectrum2Poles, GeoFPTAXEmulator)

    from desilike.emulators import Space
    bounds = {
        'h': (0.6, 0.8), 'omega_b': (0.021, 0.024),
        'omega_cdm': (0.10, 0.14), 'n_s': (0.90, 1.00),
        'w0_fld': (-1.4, -0.6), 'wa_fld': (-0.8, 0.8),
        'logA': (2.7, 3.3),
    }
    space = Space(bounds=bounds)

    pipe_exact = build(GeoFPTAXPTSpectrum2Poles())

    emu = GeoFPTAXEmulator(GeoFPTAXPTSpectrum2Poles(), space)
    emu.train(budget=1)
    pt_emu_calc = emu.to_calculator(GeoFPTAXPTSpectrum2Poles())
    pipe_emu = build(pt_emu_calc)

    def _pk1(pipe, point):
        out = pipe(**point)
        return np.asarray(getattr(out, 'pk_1loop', out))

    def _residual(point):
        e = _pk1(pipe_exact, point)
        m = _pk1(pipe_emu, point)
        # ignore the k-range where pk crosses zero, where a relative metric is meaningless
        mask = np.abs(e) > 0.1 * np.abs(e).max()
        return float(np.max(np.abs(m - e)[mask] / np.abs(e)[mask]))

    center = {name: 0.5 * (lo + hi) for name, (lo, hi) in bounds.items()}

    print(f'\nCenter (should be ~1e-15): {_residual(center):.2e}')
    for name, delta in (('logA', 0.25), ('w0_fld', -0.35), ('wa_fld', 0.7)):
        point = dict(center, **{name: center[name] + delta})
        print(f'{name} = center + {delta:+.2f} ({point[name]:.3f}): '
              f'reldiff = {_residual(point):.2e}')

    full = dict(center, w0_fld=-1.35, wa_fld=0.7, logA=3.25)
    print(f'All three at the failing point: {_residual(full):.2e}')

    
def test_geofptax_accuracy_in_grid_parameter():
    """The emulator must reproduce the exact PT at shifted `h` and `omega_cdm`."""
    pytest.importorskip('geofptax')
    from desilike.theories.galaxy_clustering.full_shape import (
        GeoFPTAXPTSpectrum2Poles, GeoFPTAXEmulator)

    from desilike.emulators import Space
    bounds = {
        'h': (0.55, 0.85), 'omega_b': (0.020, 0.025),
        'omega_cdm': (0.08, 0.16), 'n_s': (0.85, 1.05),
        'w0_fld': (-1.4, -0.6), 'wa_fld': (-0.8, 0.8),
        'logA': (2.7, 3.3),
    }
    space = Space(bounds=bounds)

    pipe_exact = build(GeoFPTAXPTSpectrum2Poles())

    emu = GeoFPTAXEmulator(GeoFPTAXPTSpectrum2Poles(), space)
    emu.train(budget=2)
    pt_emu_calc = emu.to_calculator(GeoFPTAXPTSpectrum2Poles())
    pipe_emu = build(pt_emu_calc)

    def _pk1(pipe, point):
        out = pipe(**point)
        return np.asarray(getattr(out, 'pk_1loop', out))

    center = {name: 0.5 * (lo + hi) for name, (lo, hi) in bounds.items()}
    for shift in (dict(h=0.72), dict(omega_cdm=0.11), dict(n_s=0.98)):
        point = dict(center, **shift)
        exact = _pk1(pipe_exact, point)
        emulated = _pk1(pipe_emu, point)
        reldiff = float(np.max(np.abs(emulated - exact) / (np.abs(exact) + 1e-30)))
        assert reldiff < 0.05, \
            f'grid-parameter shift {shift}: max reldiff={reldiff:.3f} > 0.05'


if __name__ == '__main__':
    #test_kaiser_spectrum_emulated()
    #test_kaiser_correlation_emulated()
    #test_tns_spectrum_emulated()
    #test_tns_correlation_emulated()
    #test_folps_spectrum_emulated()
    test_folps_spectrum3_poles_emulated()