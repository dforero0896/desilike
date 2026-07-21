"""
BAO power spectrum template for galaxy clustering.

Classes
-------
BAOSpectrum2Template
    Fiducial-cosmology BAO template: power spectra and AP distances computed once from
    cosmoprimo at compile time; scaled at evaluation time by free AP and growth-rate params.
FixedSpectrum2Template
    Fixed template: power spectrum and growth rate pinned to a fiducial cosmology, with
    no free parameters at all (no AP distortion, no growth-rate rescaling).
ShapeFitSpectrum2Template
    BAO template with ShapeFit tilt parameterisation (dm, dn).
DirectSpectrum2Template
    Direct template: power spectrum computed at every evaluation from a
    :class:`CosmoprimoCosmology` dependency.
BAOPhaseShiftSpectrum2Template
    BAO template with Baumann et al. 2018 N_eff-induced phase shift parameterisation.
TurnOverSpectrum2Template
    Template based on the matter power-spectrum turn-over scale.
DirectWiggleSplitSpectrum2Template
    Direct template with explicit wiggle/no-wiggle split and BAO dilation.
BAOExtractor
    Extracts BAO distance parameters (DH/rd, DM/rd, DV/rd, qpar, qper, qiso, qap) from a cosmology.
BAOPhaseShiftExtractor
    BAO extractor extended with the neutrino-driven BAO phase shift (N_eff, baoshift).
TurnOverExtractor
    Extracts turn-over observables (kTO, DV*kTO, DH/DM, qto, qap) from a cosmology.

Spectrum2Template contract
---------------------------
PT calculators (FOLPS, fkptjax, REPT Velocileptors, Kaiser, TNS, etc.) read template
outputs directly off ``self.template``, not through a shared base-class accessor, so
every concrete ``Spectrum2Template`` subclass must set the following attributes by the
time ``__call__`` returns (and include them in ``tree_flatten``/``tree_unflatten``):

    k, z            : output k-grid [h/Mpc] and effective redshift (set in __post_init__,
                      not __call__ -- they are fixed at compile time). By existing
                      convention ``k`` (but not ``z``) is included in ``tree_flatten``'s
                      aux dict, so it survives a tree_flatten/tree_unflatten round trip.
    pk_dd, pknow_dd : full and no-wiggle linear power spectra on `k`.
    f, f0, fk       : growth rate (sigma8-based, scale-independent), its low-k limit, and
                      its k-dependent value (from the power-spectrum ratio P_theta/P_delta).
    qpar, qper      : Alcock-Paczynski distortion ratios (line-of-sight, transverse).
    sigma8, fsigma8, sigma8_fid :
                      sigma8(z) and fsigma8(z) for the current parameters, and the
                      *fiducial* sigma8(z). Some PT classes use sigma8_fid for amplitude
                      rescaling, e.g. A = sigma8 / sigma8_fid in FOLPSTracerSpectrum2Poles.
                      If a template has no amplitude-rescaling parameter, sigma8 simply
                      equals sigma8_fid and fsigma8 = f * sigma8.
    ap_k_mu(k, mu)  : method returning (jac, kap, muap) for AP-distorting a (k, mu) grid.

"""

import numpy as np
import jax.numpy as jnp
import jax
from ...base import Calculator
from ...parameter import Parameter, VariableCollection
from ..primordial_cosmology import CosmoprimoCosmology, _get_fiducial
from ._multitracer import propose_params_multitracer, assign_params


_kw_pk = dict(extrap_kmin=1e-7, extrap_kmax=1e2)  # cosmoprimo pk_interpolator kwargs


# ── Base class ────────────────────────────────────────────────────────────────

class Spectrum2Template(Calculator):
    """Marker base class for all 2-point power-spectrum template calculators.

    Subclassed by :class:`BAOSpectrum2Template`, :class:`ShapeFitSpectrum2Template`,
    and :class:`DirectSpectrum2Template` so that code can use ``isinstance`` checks
    rather than duck-typing on constructor kwargs.
    """


# ── AP distortion ─────────────────────────────────────────────────────────────

def _ap_k_mu(k, mu, qpar, qper):
    """Alcock-Paczynski distortion of (k, mu) grid.

    k   : (...,) or broadcastable
    mu  : (...,) or broadcastable
    Returns (jac, kap, muap) of same broadcast shape.
    """
    qap = qpar / qper
    jac = 1. / (qpar * qper**2)
    factorap = jnp.sqrt(1. + mu**2 * (1. / qap**2 - 1.))
    kap = k / qper * factorap
    muap = mu / qap / factorap
    return jac, kap, muap


# ── shared helper ─────────────────────────────────────────────────────────────

def _ap_auto_params(apmode):
    """Return the AP Parameter list for the given *apmode*.  Raises on unknown mode."""
    _ap_prior = dict(limits=[0.5, 2.])
    _ap_ref = dict(dist='norm', loc=1., scale=0.05)
    _ap_fd = 0.008
    if apmode == 'qparqper':
        return [Parameter('qpar', value=1., prior=_ap_prior, ref=_ap_ref, fd_eps=_ap_fd, latex=r'q_\parallel'),
                Parameter('qper', value=1., prior=_ap_prior, ref=_ap_ref, fd_eps=_ap_fd, latex=r'q_\perp')]
    if apmode == 'qisoqap':
        return [Parameter('qiso', value=1., prior=_ap_prior, ref=_ap_ref, fd_eps=_ap_fd, latex=r'q_\mathrm{iso}'),
                Parameter('qap', value=1., prior=_ap_prior, ref=_ap_ref, fd_eps=_ap_fd, latex=r'q_\mathrm{ap}')]
    if apmode == 'qiso':
        return [Parameter('qiso', value=1., prior=_ap_prior, ref=_ap_ref, fd_eps=_ap_fd, latex=r'q_\mathrm{iso}')]
    if apmode == 'qap':
        return [Parameter('qap', value=1., prior=_ap_prior, ref=_ap_ref, fd_eps=_ap_fd, latex=r'q_\mathrm{ap}')]
    raise ValueError(f"apmode must be one of 'qparqper', 'qisoqap', 'qiso', 'qap'; got {apmode!r}")

def _integrate_sigma_r2_jax(r, pk, k_fine):
    """JAX-traceable version for runtime evaluation."""
    logk = jnp.log(k_fine)
    x = k_fine * r
    small = x < 1e-2
    w2_small = 1.0 - x**2 / 5.0
    w2_large = 9.0 * (jnp.sin(x) - x * jnp.cos(x))**2 / x**6
    w2 = jnp.where(small, w2_small, w2_large)
    integrand = pk * w2 * k_fine**3
    dlogk = logk[1:] - logk[:-1]
    integral = jnp.sum(0.5 * (integrand[1:] + integrand[:-1]) * dlogk)
    return 1. / (2. * jnp.pi**2) * integral

def _compute_shapefit_fiducials(fiducial_cosmo, z, kp, a, with_now, r=8., n_varied=False):
    """Compute ShapeFit fiducial quantities at compile time using native cosmoprimo APIs."""
    from cosmoprimo import PowerSpectrumBAOFilter
    fo = fiducial_cosmo.get_fourier()
    
    # 1. Directly use cosmoprimo for sigma8 and fsigma8 (exactly like the template)
    sigma8_fid = float(fo.sigma8_z(z, of='delta_cb'))
    fsigma8_fid = float(fo.sigma8_z(z, of='theta_cb')) # This is effectively f * sigma8
    f_fid = fsigma8_fid / sigma8_fid
    n_fid = float(fiducial_cosmo.n_s)
    # 2. Get power spectrum interpolators
    pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=z)
    if with_now:
        bao_filter = PowerSpectrumBAOFilter(pk_interp, engine=with_now, cosmo=fiducial_cosmo, cosmo_fid=fiducial_cosmo)
        pknow_interp = bao_filter.smooth_pk_interpolator()
    else:
        pknow_interp = pk_interp
        
    # 3. Compute slope 'm' at pivot kp using finite differences
    dk = 1e-2
    k_query = kp * np.array([1. - dk, 1., 1. + dk])
    pknow_minus = float(pknow_interp(k_query[0]))
    pknow_kp = float(pknow_interp(k_query[1]))
    pknow_plus = float(pknow_interp(k_query[2]))
    
    slope_pknow = (np.log(pknow_plus) - np.log(pknow_minus)) / (np.log(k_query[2]) - np.log(k_query[0]))
    slope_pk_prim = (n_fid + 1.0) if n_varied else 0.0
    m_fid = slope_pknow - slope_pk_prim
    
    # 4. Amplitude Ap
    Ap_fid = pknow_kp
    f_sqrt_Ap_fid = f_fid * Ap_fid**0.5
    
    # 5. Directly use cosmoprimo for sigmar and fsigmar at the fixed fiducial radius r
    # (If r=8, this is mathematically identical to sigma8_fid)
    sigmar_fid = float(fo.sigma_rz(r, z, of='delta_cb'))
    fsigmar_fid = float(fo.sigma_rz(r, z, of='theta_cb'))
    
    return {
        'm_fid': m_fid, 'Ap_fid': Ap_fid, 'f_sqrt_Ap_fid': f_sqrt_Ap_fid,
        'f_sigmar_fid': fsigmar_fid, 'n_fid': n_fid, 'f_fid': f_fid, 
        'sigma8_fid': sigma8_fid, 'fsigma8_fid': fsigma8_fid, 
        'sigmar_fid': sigmar_fid
    }

# ── BAO template ──────────────────────────────────────────────────────────────

class BAOSpectrum2Template(Spectrum2Template):
    r"""
    BAO power spectrum template based on a fixed fiducial cosmology.

    The fiducial power spectra, growth rates, and BAO distances are computed once from
    cosmoprimo at compile time (``__post_init__``). At evaluation time (``__call__``),
    power spectra are copied from fiducial arrays and the growth rate and distances are
    scaled by the free parameters.

    Parameters
    ----------
    k : array, default=None
        Wavenumbers [h/Mpc]. Defaults to np.logspace(-3, 1, 400).
    z : float, default=1.
        Effective redshift.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology. A string is looked up as ``cosmoprimo.fiducial.<name>()``.
    with_now : str or False, default='peakaverage'
        Engine for the BAO-filtered smooth power spectrum ('peakaverage', 'wallish2018').
        Set to False to skip (pknow_dd is set equal to pk_dd).
    only_now : bool, default=False
        Replace pk_dd with pknow_dd so wiggles are absent from the model.
    apmode : str, default='qparqper'
        AP parameterization. One of:

        - 'qparqper': free parameters ``qpar`` (LOS scaling) and ``qper`` (transverse scaling).
        - 'qisoqap':  free parameters ``qiso`` and ``qap = qpar / qper``.
        - 'qiso':     single isotropic parameter ``qiso``.
        - 'qap':      single AP parameter ``qap``.
    eta : float, default=1./3.
        Exponent in  qiso = qpar**eta * qper**(1 - eta).

    Attributes set by ``__call__``
    --------------------------------
    pk_dd, pknow_dd : ndarray, shape (n_k,)
        Full and smooth (no-wiggle) power spectra at ``self.k``.
    f, f0, fk : float or ndarray
        Growth rate f = d ln D / d ln a;  f0 is the k->0 limit, fk is k-dependent.
    qpar, qper : float
        AP distortion ratios, derived from the sampled apmode parameters.
    sigma8, fsigma8, sigma8_fid : float
        Fixed at the fiducial value (no amplitude-rescaling parameter); fsigma8 tracks
        the df-scaled growth rate.
    DH_over_rd, DM_over_rd, DV_over_rd : float
        BAO distance ratios scaled by the AP parameters.
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

    @classmethod
    def propose_params(cls, apmode='qparqper'):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this template.

        Parameters
        ----------
        apmode : str, default='qparqper'
            AP parameterization: one of ``'qparqper'``, ``'qisoqap'``, ``'qiso'``, ``'qap'``.

        Returns
        -------
        VariableCollection
        """
        return propose_params_multitracer(
            _ap_auto_params(apmode) + [
                Parameter('df', value=1., fixed=True, prior=dict(limits=[0., 2.]),
                          ref=dict(dist='norm', loc=1., scale=0.05), fd_eps=0.02, latex=r'\delta f'),
            ], tracers=None)

    def __init__(self, k=None, z=1., fiducial='DESI', with_now='peakaverage',
                 only_now=False, apmode='qparqper', eta=1. / 3., params=None):
        # AP parameters — created here so they appear in __dict__ for graph scan.
        vc = type(self).propose_params(apmode=str(apmode))
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, None)
        # self.params keeps the apmode Parameters reachable by name, independent of the
        # public self.qpar/self.qper attribute, which __call__ reassigns to the derived
        # plain output value (see the module docstring contract). _qpar_qper() reads from
        # self.params rather than self.qpar/self.qper so it survives that reassignment.
        self.params = vc

    def __post_init__(self, k=None, z=1., fiducial='DESI', with_now='peakaverage',
                      only_now=False, apmode='qparqper', eta=1. / 3., params=None):
        from cosmoprimo import PowerSpectrumBAOFilter, constants

        self._apmode = str(apmode)
        self._eta = float(eta)
        self._only_now = bool(only_now)

        if k is None:
            k = np.logspace(-3., 1., 400)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)

        self._fiducial = _get_fiducial(fiducial)

        fo = self._fiducial.get_fourier()
        sigma8 = fo.sigma8_z(z, of='delta_cb')
        fsigma8 = fo.sigma8_z(z, of='theta_cb')
        self._sigma8_fid = float(sigma8)
        self._f_fid = float(fsigma8 / sigma8)

        pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=z)
        ptt_interp = fo.pk_interpolator(of='theta_cb', **_kw_pk).to_1d(z=z)

        k0 = 1e-3  # low-k limit for f0
        self._f0_fid = float(np.sqrt(ptt_interp(k0) / pk_interp(k0)))
        self._fk_fid = np.sqrt(ptt_interp(self.k) / pk_interp(self.k))
        self._pk_dd_fid = pk_interp(self.k)

        if with_now:
            bao_filter = PowerSpectrumBAOFilter(pk_interp, engine=with_now, cosmo=self._fiducial, cosmo_fid=self._fiducial)
            self._pknow_dd_fid = bao_filter.smooth_pk_interpolator()(self.k)
        else:
            self._pknow_dd_fid = self._pk_dd_fid

        # Fiducial BAO distance ratios
        rd = self._fiducial.rs_drag
        DH_fid = constants.c / 1e3 / (100. * self._fiducial.efunc(z))
        DM_fid = self._fiducial.comoving_transverse_distance(z)
        DV_fid = DH_fid**eta * DM_fid**(1. - eta) * z**(1. / 3.)
        self._DH_over_rd_fid = float(DH_fid / rd)
        self._DM_over_rd_fid = float(DM_fid / rd)
        self._DV_over_rd_fid = float(DV_fid / rd)
        self.sigma8_fid = jnp.asarray(self._sigma8_fid)

    def _qpar_qper(self):
        """Convert current apmode parameter values to (qpar, qper)."""
        if self._apmode == 'qparqper':
            return self.params['qpar'].value, self.params['qper'].value
        if self._apmode == 'qiso':
            q = self.params['qiso'].value
            return q, q
        if self._apmode == 'qap':
            qap = self.params['qap'].value
            return qap ** (1. - self._eta), qap ** (-self._eta)
        # qisoqap
        qiso, qap = self.params['qiso'].value, self.params['qap'].value
        return qiso * qap ** (1. - self._eta), qiso * qap ** (-self._eta)

    def ap_k_mu(self, k, mu):
        """Apply AP distortion to a (k, mu) grid; returns (jac, kap, muap)."""
        qpar, qper = self._qpar_qper()
        return _ap_k_mu(k, mu, qpar, qper)

    def __call__(self):
        # Power spectra: fixed at fiducial (no cosmo call at eval time).
        self.pk_dd = self._pk_dd_fid
        self.pknow_dd = self._pknow_dd_fid
        if self._only_now:
            self.pk_dd = self._pknow_dd_fid

        # Growth rate scaled by df.
        df = self.df.value
        self.f = self._f_fid * df
        self.f0 = self._f0_fid * df
        self.fk = self._fk_fid * df

        # AP parameters and BAO distances, both derived from the sampled apmode params.
        qpar, qper = self._qpar_qper()
        self.qpar = qpar
        self.qper = qper
        self.DH_over_rd = qpar * self._DH_over_rd_fid
        self.DM_over_rd = qper * self._DM_over_rd_fid
        self.DV_over_rd = qpar ** self._eta * qper ** (1. - self._eta) * self._DV_over_rd_fid

        # No amplitude-rescaling parameter: sigma8 stays at its fiducial value;
        # fsigma8 tracks the df-scaled growth rate.
        self.sigma8 = jnp.asarray(self._sigma8_fid)
        self.fsigma8 = self.f * self.sigma8
        self.sigma8_fid = jnp.asarray(self._sigma8_fid)

        return self.pk_dd

    def tree_flatten(self):
        return ([self.pk_dd, self.pknow_dd, self.f, self.f0, self.fk, self.qpar, self.qper,
                 self.sigma8, self.fsigma8, self.sigma8_fid], {'k': self.k})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (obj.pk_dd, obj.pknow_dd, obj.f, obj.f0, obj.fk, obj.qpar, obj.qper,
         obj.sigma8, obj.fsigma8, obj.sigma8_fid) = children
        obj.k = aux['k']
        return obj


# ── Fixed template ────────────────────────────────────────────────────────────

class FixedSpectrum2Template(Spectrum2Template):
    r"""
    Fixed power spectrum template.

    Power spectrum and growth rate are pinned to a fixed fiducial cosmology, with
    no free parameters at all: no Alcock-Paczynski distortion (``qpar = qper = 1``)
    and no growth-rate rescaling. Useful e.g. for forecasts/validation against a
    fiducial cosmology, or as a placeholder template when AP/growth-rate freedom is
    not wanted.

    Parameters
    ----------
    k : array, default=None
        Wavenumbers [h/Mpc]. Defaults to np.logspace(-3, 1, 400).
    z : float, default=1.
        Effective redshift.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology. A string is looked up as ``cosmoprimo.fiducial.<name>()``.
    with_now : str or False, default='peakaverage'
        Engine for the BAO-filtered smooth power spectrum ('peakaverage', 'wallish2018').
        Set to False to skip (pknow_dd is set equal to pk_dd).
    only_now : bool, default=False
        Replace pk_dd with pknow_dd so wiggles are absent from the model.

    Attributes set by ``__call__``
    --------------------------------
    pk_dd, pknow_dd : ndarray, shape (n_k,)
        Full and smooth (no-wiggle) power spectra at ``self.k``, fixed to the fiducial cosmology.
    f, f0, fk : float or ndarray
        Growth rate, fixed to the fiducial cosmology.
    qpar, qper : float
        Always 1 (no AP distortion).
    sigma8, fsigma8, sigma8_fid : float
        Fixed at the fiducial value (no amplitude-rescaling parameter); fsigma8 = f * sigma8.
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

    @classmethod
    def propose_params(cls):
        """No free parameters at all."""
        return VariableCollection()

    def __init__(self, k=None, z=1., fiducial='DESI', engine='class', with_now='peakaverage', only_now=False):
        self._fiducial = CosmoprimoCosmology(engine=engine, fiducial=fiducial)
        _get_fiducial(fiducial, calculator=self._fiducial)  # runs _fiducial at fiducial params (sets _cosmo)

    @property
    def cosmo(self):
        return self._fiducial

    def __post_init__(self, k=None, z=1., fiducial='DESI', engine='class', with_now='peakaverage', only_now=False):
        from cosmoprimo import PowerSpectrumBAOFilter

        self._only_now = bool(only_now)

        if k is None:
            k = np.logspace(-3., 1., 400)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)

        _cosmo = self._fiducial._cosmo

        fo = _cosmo.get_fourier()
        sigma8 = fo.sigma8_z(z, of='delta_cb')
        fsigma8 = fo.sigma8_z(z, of='theta_cb')
        self._sigma8_fid = float(sigma8)
        self._f_fid = float(fsigma8 / sigma8)

        pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=z)
        ptt_interp = fo.pk_interpolator(of='theta_cb', **_kw_pk).to_1d(z=z)

        k0 = 1e-3  # low-k limit for f0
        self._f0_fid = float(np.sqrt(ptt_interp(k0) / pk_interp(k0)))
        self._fk_fid = np.sqrt(ptt_interp(self.k) / pk_interp(self.k))
        self._pk_dd_fid = pk_interp(self.k)

        if with_now:
            bao_filter = PowerSpectrumBAOFilter(pk_interp, engine=with_now, cosmo=_cosmo, cosmo_fid=_cosmo)
            self._pknow_dd_fid = bao_filter.smooth_pk_interpolator()(self.k)
        else:
            self._pknow_dd_fid = self._pk_dd_fid

    def ap_k_mu(self, k, mu):
        """No AP distortion: identity transform (jac=1, k and mu unchanged)."""
        return _ap_k_mu(k, mu, self.qpar, self.qper)

    def __call__(self):
        self.pk_dd = self._pk_dd_fid
        self.pknow_dd = self._pknow_dd_fid
        if self._only_now:
            self.pk_dd = self._pknow_dd_fid

        self.f = self._f_fid
        self.f0 = self._f0_fid
        self.fk = self._fk_fid
        self.qpar = 1.
        self.qper = 1.
        # No amplitude-rescaling parameter: sigma8 stays at its fiducial value.
        self.sigma8 = jnp.asarray(self._sigma8_fid)
        self.fsigma8 = self.f * self.sigma8
        self.sigma8_fid = jnp.asarray(self._sigma8_fid)

        return self.pk_dd

    def tree_flatten(self):
        return ([self.pk_dd, self.pknow_dd, self.f, self.f0, self.fk, self.qpar, self.qper,
                 self.sigma8, self.fsigma8, self.sigma8_fid], {'k': self.k})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (obj.pk_dd, obj.pknow_dd, obj.f, obj.f0, obj.fk, obj.qpar, obj.qper,
         obj.sigma8, obj.fsigma8, obj.sigma8_fid) = children
        obj.k = aux['k']
        return obj


class ShapeFitSpectrum2Template(Spectrum2Template):
    r"""
    ShapeFit power spectrum template.

    Multiplies the fiducial power spectrum by a k-dependent tilt factor controlled by ``dm`` and ``dn``.

    Parameters
    ----------
    k : array, default=None
        Wavenumbers [h/Mpc]. Defaults to np.logspace(-3, 1, 400).
    z : float, default=1.
        Effective redshift.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology.
    with_now : str or False, default='peakaverage'
        Engine for the no-wiggle power spectrum ('peakaverage', 'wallish2018').
        Set to False to skip (pknow_dd is set equal to pk_dd).
    only_now : bool, default=False
        Replace pk_dd with pknow_dd so wiggles are absent.
    apmode : str, default='qparqper'
        AP parameterization: 'qparqper', 'qisoqap', 'qiso', or 'qap'.
    eta : float, default=1./3.
        Exponent in qiso = qpar**eta * qper**(1 - eta).
    kp : float, default=0.03
        Pivot wavenumber [h/Mpc] for the ShapeFit parameterization.
    a : float, default=0.6
        Steepness parameter in the ShapeFit tilt function.

    Attributes set by ``__call__``
    --------------------------------
    pk_dd, pknow_dd : ndarray, shape (n_k,)
        Full and smooth (no-wiggle) power spectra.
    f, f0, fk : float or ndarray
        Growth rate.
    qpar, qper : float
        AP distortion ratios, derived from the sampled apmode parameters.
    sigma8, fsigma8, sigma8_fid : float
        sigma8 stays at its fiducial value (no amplitude-rescaling parameter); fsigma8
        tracks the df-scaled growth rate.
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

    @classmethod
    def propose_params(cls, apmode='qparqper'):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for this template.

        Parameters
        ----------
        apmode : str, default='qparqper'
            AP parameterization: one of ``'qparqper'``, ``'qisoqap'``, ``'qiso'``, ``'qap'``.

        Returns
        -------
        VariableCollection
        """
        return propose_params_multitracer(
            _ap_auto_params(apmode) + [
                Parameter('df', value=1., prior=dict(limits=[0., 2.]),
                          ref=dict(dist='norm', loc=1., scale=0.05), fd_eps=0.02, latex=r'\delta f'),
                Parameter('dm', value=0., prior=dict(limits=[-0.5, 0.5]),
                          ref=dict(dist='norm', loc=0., scale=0.05), fd_eps=0.01, latex=r'\delta m'),
                Parameter('dn', value=0., fixed=True, prior=dict(limits=[-0.5, 0.5]),
                          ref=dict(dist='norm', loc=0., scale=0.05), fd_eps=0.01, latex=r'\delta n'),
                Parameter('dAp', value=1., prior=dict(limits=[0., 2.]),
                          ref=dict(dist='norm', loc=1., scale=0.05), fd_eps=0.02, latex=r'\delta A_{p}'),
            ], tracers=None)

    def __init__(self, k=None, z=1., fiducial='DESI', with_now='peakaverage',
                 only_now=False, apmode='qparqper', eta=1. / 3., kp=0.03, a=0.6, params=None,
                 engine = 'class', cosmo = None):
        vc = type(self).propose_params(apmode=str(apmode))
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, None)
        # See BAOSpectrum2Template.__init__: _qpar_qper() reads from self.params rather
        # than self.qpar/self.qper, since __call__ reassigns those to the derived output.
        self.params = vc

        if cosmo is None:
            cosmo = CosmoprimoCosmology(engine=engine, fiducial=fiducial)
        self.cosmo = cosmo

    def __post_init__(self, k=None, z=1., fiducial='DESI', with_now='peakaverage',
                      only_now=False, apmode='qparqper', eta=1. / 3., kp=0.03, a=0.6, params=None):
        from cosmoprimo import PowerSpectrumBAOFilter

        self._apmode = str(apmode)
        self._eta = float(eta)
        self._only_now = bool(only_now)
        self._kp = float(kp)
        self._a = float(a)

        if k is None:
            k = np.logspace(-3., 1., 400)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)

        self._fiducial = _get_fiducial(fiducial)
        self._rs_drag_fid = self._fiducial.rs_drag
        fo = self._fiducial.get_fourier()
        sigma8 = fo.sigma8_z(z, of='delta_cb')
        fsigma8 = fo.sigma8_z(z, of='theta_cb')
        self._sigma8_fid = float(sigma8)
        self._fsigma8_fid = float(fsigma8)
        self._f_fid = float(fsigma8 / sigma8)

        pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=z)
        ptt_interp = fo.pk_interpolator(of='theta_cb', **_kw_pk).to_1d(z=z)

        k0 = 1e-3
        self._f0_fid = float(np.sqrt(ptt_interp(k0) / pk_interp(k0)))
        self._fk_fid = np.sqrt(ptt_interp(self.k) / pk_interp(self.k))
        self._pk_dd_fid = pk_interp(self.k)

        if with_now:
            bao_filter = PowerSpectrumBAOFilter(pk_interp, engine=with_now, cosmo=self._fiducial, cosmo_fid=self._fiducial)
            self._pknow_dd_fid = bao_filter.smooth_pk_interpolator()(self.k)
        else:
            self._pknow_dd_fid = self._pk_dd_fid
        self.sigma8_fid = jnp.asarray(self._sigma8_fid)

        fiducials = _compute_shapefit_fiducials(
            self._fiducial, self.z, self._kp, self._a, with_now, n_varied=False
        )
        self._Ap_fid = float(fiducials['Ap_fid'])
        self._m_fid = float(fiducials['m_fid'])
        self._n_fid = float(fiducials['n_fid'])
        self._sigmar_fid = float(fiducials['sigmar_fid'])

    def _qpar_qper(self):
        if self._apmode == 'qparqper':
            return self.params['qpar'].value, self.params['qper'].value
        if self._apmode == 'qiso':
            q = self.params['qiso'].value
            return q, q
        if self._apmode == 'qap':
            qap = self.params['qap'].value
            return qap ** (1. - self._eta), qap ** (-self._eta)
        qiso, qap = self.params['qiso'].value, self.params['qap'].value
        return qiso * qap ** (1. - self._eta), qiso * qap ** (-self._eta)

    def ap_k_mu(self, k, mu):
        """Apply AP distortion to a (k, mu) grid; returns (jac, kap, muap)."""
        qpar, qper = self._qpar_qper()
        return _ap_k_mu(k, mu, qpar, qper)

    def __call__(self):
        dm = self.dm.value
        dn = self.dn.value
        df = self.df.value
        dAp = self.dAp.value
        factor = jnp.exp(dm / self._a * jnp.tanh(self._a * jnp.log(self.k / self._kp))
                         + dn * jnp.log(self.k / self._kp))
        self.pk_dd = dAp * self._pk_dd_fid * factor
        self.pknow_dd = dAp * self._pknow_dd_fid * factor
        if self._only_now:
            self.pk_dd = self.pknow_dd
        self.f = self._f_fid * df
        self.f0 = self._f0_fid * df
        self.fk = self._fk_fid * df
        qpar, qper = self._qpar_qper()
        self.qpar = qpar
        self.qper = qper
        # Update sigma8 & fsigma8 using Eq. A.12 ---
        # (sigma_s8 / sigma_s8_fid)^2 = dAp * exp((dm + dn)/a * tanh(a * ln(r_d_fid / 8)))
        tanh_arg = self._a * jnp.log(self._rs_drag_fid / 8.)
        dsigma8 = dAp * jnp.exp((dm + dn) / self._a * jnp.tanh(tanh_arg))
        
        self.sigma8 = self._sigma8_fid * jnp.sqrt(dsigma8)
        self.fsigma8 = self.f * self.sigma8
        # ---------------------------------------------------------
        
        self.sigma8_fid = jnp.asarray(self._sigma8_fid)


         # --- NEW: Compute and expose physical ShapeFit parameters ---
        self.Ap = dAp * self._Ap_fid
        self.m = self._m_fid + dm
        self.n = self._n_fid + dn
        self.f_sqrt_Ap = self.f * self.Ap**0.5
        # ----------------------------------------------------------

        return self.pk_dd

    def tree_flatten(self):
        # Add the new ShapeFit parameters to the JAX tree leaves
        return ([self.pk_dd, self.pknow_dd, self.f, self.f0, self.fk, self.qpar, self.qper,
                 self.sigma8, self.fsigma8, self.sigma8_fid,
                 self.Ap, self.m, self.n, self.f_sqrt_Ap], 
                {'k': self.k})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        # Unflatten the new leaves
        (obj.pk_dd, obj.pknow_dd, obj.f, obj.f0, obj.fk, obj.qpar, obj.qper,
         obj.sigma8, obj.fsigma8, obj.sigma8_fid,
         obj.Ap, obj.m, obj.n, obj.f_sqrt_Ap) = children
        obj.k = aux['k']
        return obj


class DirectSpectrum2Template(Spectrum2Template):
    r"""
    Direct power spectrum template: power spectrum evaluated at each pipeline call from a
    :class:`CosmoprimoCosmology` dependency.

    AP parameters (qpar, qper) are computed from the ratio of current to fiducial distances.
    By default a :class:`CosmoprimoCosmology` calculator is created internally; an existing
    instance may be passed via ``cosmo`` to share cosmological parameters across theories.

    Parameters
    ----------
    k : array, default=None
        Wavenumbers [h/Mpc]. Defaults to ``np.logspace(-3, 1, 400)``.
    z : float, default=1.
        Effective redshift.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology for AP ratio denominator and fiducial PK/no-wiggle PK.
    engine : str, default='camb'
        Boltzmann solver engine forwarded to the internal :class:`CosmoprimoCosmology`
        (ignored when ``cosmo`` is supplied).
    with_now : str or False, default=False
        No-wiggle filter engine ('peakaverage', 'wallish2018'); ``False`` to skip.
    only_now : bool, default=False
        Replace ``pk_dd`` with ``pknow_dd`` so wiggles are absent.
    cosmo : CosmoprimoCosmology or None, default=None
        External cosmology calculator to use as a dep.  When ``None`` a fresh
        :class:`CosmoprimoCosmology` is created with the given ``engine`` and
        ``fiducial`` defaults.

    Attributes set by ``__call__``
    --------------------------------
    pk_dd, pknow_dd : ndarray, shape (n_k,)
        Full and smooth (no-wiggle) power spectra.
    f, f0, fk : float or ndarray
        Growth rate.
    sigma8, fsigma8 : float
        Normalisation and growth-rate normalisation.
    qpar, qper : float
        AP distortion ratios (current / fiducial distances).
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

    @classmethod
    def propose_params(cls, engine='class', fiducial=None):
        """Return a proposed :class:`~desilike.parameter.VariableCollection` for the cosmological parameters.

        Delegates to :meth:`~desilike.theories.primordial_cosmology.CosmoprimoCosmology.propose_params`.

        Parameters
        ----------
        engine : str, default='camb'
        fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default=None

        Returns
        -------
        VariableCollection
        """
        return CosmoprimoCosmology.propose_params(engine=engine, fiducial=fiducial)

    def __init__(self, k=None, z=1., fiducial='DESI', engine='class', with_now=False, only_now=False, cosmo=None):
        if cosmo is None:
            cosmo = CosmoprimoCosmology(engine=engine, fiducial=fiducial)
        self.cosmo = cosmo

    def __post_init__(self, k=None, z=1., fiducial='DESI', engine='class', with_now=False, only_now=False, cosmo=None):
        # Non-node setup: fiducial distances and fiducial PK (fixed at compile time).
        from cosmoprimo import PowerSpectrumBAOFilter, constants
        if k is None:
            k = np.logspace(-3., 1., 400)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)
        self._with_now = with_now
        self._only_now = bool(only_now)
        # Prepend k0 = 1e-3 so get_result(...)[0] gives pk at k0 for f0 = sqrt(ptt/pk)|_{k→0}.
        self._k_with_k0 = np.concatenate([[1e-3], self.k])
        reqs = {
            'fourier.pk': [
                {'of': 'delta_cb', 'z': self.z, 'k': self._k_with_k0},
                {'of': 'theta_cb', 'z': self.z, 'k': self._k_with_k0},
            ],
            'fourier.sigma8_z': [
                {'of': 'delta_cb', 'z': self.z},
                {'of': 'theta_cb', 'z': self.z},
            ],
            'background.efunc':                        [{'z': self.z}],
            'background.comoving_transverse_distance': [{'z': self.z}],
        }
        if with_now:
            reqs['fourier.pk_now'] = [
                {'of': 'delta_cb', 'engine': str(with_now), 'z': self.z, 'k': self.k},
            ]
        self.cosmo.add_requirements(reqs)

        self._fiducial = _get_fiducial(fiducial)
        self._DH_fid = float(constants.c / 1e3 / (100. * self._fiducial.efunc(self.z)))
        self._DM_fid = float(self._fiducial.comoving_transverse_distance(self.z))

        # Fiducial PK arrays (used by e.g. ResummedBAOWigglesPTSpectrum2Poles for damping scales).
        fo = self._fiducial.get_fourier()
        self._sigma8_fid = float(fo.sigma8_z(self.z, of='delta_cb'))
        pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=self.z)
        self._pk_dd_fid = pk_interp(self.k)
        if with_now:
            bao_filter = PowerSpectrumBAOFilter(pk_interp, engine=with_now, cosmo=self._fiducial, cosmo_fid=self._fiducial)
            self._pknow_dd_fid = bao_filter.smooth_pk_interpolator()(self.k)
        else:
            self._pknow_dd_fid = self._pk_dd_fid
        self.sigma8_fid = jnp.asarray(self._sigma8_fid)

    def __call__(self):
        from cosmoprimo import constants
        # All cosmoprimo work happened in CosmoprimoCosmology.__call__; retrieve JAX arrays.
        pk_full  = self.cosmo.get_fourier().pk(of='delta_cb', z=self.z, k=self._k_with_k0)
        ptt_full = self.cosmo.get_fourier().pk(of='theta_cb', z=self.z, k=self._k_with_k0)
        self.pk_dd = pk_full[1:]
        self.pknow_dd = (self.cosmo.get_fourier().pk_now(of='delta_cb',
                              engine=self._with_now, z=self.z, k=self.k)
                         if self._with_now else self.pk_dd)
        if self._only_now:
            self.pk_dd = self.pknow_dd
        self.sigma8  = self.cosmo.get_fourier().sigma8_z(of='delta_cb', z=self.z)
        self.fsigma8 = self.cosmo.get_fourier().sigma8_z(of='theta_cb', z=self.z)
        self.f  = self.fsigma8 / self.sigma8
        self.f0 = jnp.sqrt(ptt_full[0] / pk_full[0])   # k0 = 1e-3 is index 0
        self.fk = jnp.sqrt(ptt_full[1:] / pk_full[1:])
        DH = constants.c / 1e3 / (100. * self.cosmo.get_background().efunc(z=self.z))
        DM = self.cosmo.get_background().comoving_transverse_distance(z=self.z)
        self.qpar = DH / self._DH_fid
        self.qper = DM / self._DM_fid
        self.sigma8_fid = jnp.asarray(self._sigma8_fid)
        return self.pk_dd

    def ap_k_mu(self, k, mu):
        """Apply AP distortion; works in JAX context after tree_unflatten."""
        return _ap_k_mu(k, mu, self.qpar, self.qper)

    def tree_flatten(self):
        return ([self.pk_dd, self.pknow_dd, self.f, self.f0, self.fk,
                 self.qpar, self.qper, self.sigma8, self.fsigma8, self.sigma8_fid],
                {'k': self.k})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.pk_dd, obj.pknow_dd, obj.f, obj.f0, obj.fk, obj.qpar, obj.qper, obj.sigma8, obj.fsigma8, obj.sigma8_fid = children
        obj.k = aux['k']
        return obj


# ── BAO phase shift template ──────────────────────────────────────────────────

class BAOPhaseShiftSpectrum2Template(BAOSpectrum2Template):
    r"""
    BAO power spectrum template with an :math:`N_\mathrm{eff}`-induced phase shift.

    Extends :class:`BAOSpectrum2Template` by applying a scale-dependent phase shift to
    the BAO wiggles, following Baumann et al. 2018 (https://arxiv.org/pdf/1803.10741).
    The shift amplitude profile is

    .. math::
        k_\mathrm{shift}(k) = \frac{\phi_\infty}{1 + (k_*/k)^\epsilon} \, / \, r_\mathrm{drag}

    and the wiggles at each :math:`k` are evaluated at the shifted position
    :math:`k + (\texttt{baoshift} - 1) \cdot k_\mathrm{shift}(k)`.

    Parameters
    ----------
    phiinf : float, default=0.227
    kstar : float, default=0.0324
    epsilon : float, default=0.872
        Phase-shift profile parameters (best-fit from Baumann et al. 2018).
    (All other parameters as in :class:`BAOSpectrum2Template`.)

    Additional free parameter
    -------------------------
    baoshift : float, default=1.
        BAO phase-shift amplitude.  ``baoshift = 1`` is no shift.
    """

    @classmethod
    def propose_params(cls, apmode='qparqper'):
        """Return a proposed parameter collection including ``baoshift``."""
        return super().propose_params(apmode=apmode) + VariableCollection([
            Parameter('baoshift', value=1., prior=dict(limits=[0., 2.]),
                      ref=dict(dist='norm', loc=1., scale=0.1),
                      fd_eps=0.01, latex=r'\phi_\mathrm{BAO}'),
        ])

    def __init__(self, k=None, z=1., fiducial='DESI', with_now='peakaverage',
                 only_now=False, apmode='qparqper', eta=1. / 3.,
                 phiinf=0.227, kstar=0.0324, epsilon=0.872, params=None):
        vc = type(self).propose_params(apmode=str(apmode))
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, None)
        # See BAOSpectrum2Template.__init__: _qpar_qper() reads from self.params rather
        # than self.qpar/self.qper, since __call__ reassigns those to the derived output.
        self.params = vc

    def __post_init__(self, k=None, z=1., fiducial='DESI', with_now='peakaverage',
                      only_now=False, apmode='qparqper', eta=1. / 3.,
                      phiinf=0.227, kstar=0.0324, epsilon=0.872, params=None):
        from cosmoprimo import PowerSpectrumBAOFilter
        super().__post_init__(k=k, z=z, fiducial=fiducial, with_now=with_now,
                              only_now=only_now, apmode=apmode, eta=eta, params=params)
        self._phiinf = float(phiinf)
        self._kstar = float(kstar)
        self._epsilon = float(epsilon)
        # Dense k grid for wiggle (pk - pknow) interpolation under the BAO shift.
        k_fine = np.geomspace(_kw_pk['extrap_kmin'], _kw_pk['extrap_kmax'], 2000)
        fo = self._fiducial.get_fourier()
        pk1d = fo.pk_interpolator(of='delta_cb', **_kw_pk).to_1d(z=float(z))
        bao_filter = PowerSpectrumBAOFilter(pk1d, engine=str(with_now), cosmo=self._fiducial, cosmo_fid=self._fiducial)
        self._k_fine = k_fine
        self._wiggles_fine = pk1d(k_fine) - bao_filter.smooth_pk_interpolator()(k_fine)

    def __call__(self):
        super().__call__()  # sets pk_dd, pknow_dd, f, f0, fk, DH_over_rd, etc.
        baoshift = self.baoshift.value
        kshift = self._phiinf / (1. + (self._kstar / self.k) ** self._epsilon) / self._fiducial.rs_drag
        k_shifted = jnp.clip(self.k + (baoshift - 1.) * kshift, self._k_fine[0], self._k_fine[-1])
        wiggles = jnp.interp(jnp.log10(k_shifted), jnp.log10(self._k_fine), self._wiggles_fine)
        self.pk_dd = self._pknow_dd_fid + wiggles
        if self._only_now:
            self.pk_dd = self.pknow_dd
        return self.pk_dd


# ── Turn-over template ────────────────────────────────────────────────────────

def _find_turn_over(k, pk):
    """Locate the turn-over of *pk* on grid *k* using parabolic interpolation."""
    imax = int(np.argmax(pk))
    logk = np.log10(k[imax - 1:imax + 2])
    logpk = np.log10(pk[imax - 1:imax + 2])
    c0 = logpk[0] / ((logk[0] - logk[1]) * (logk[0] - logk[2]))
    c1 = logpk[1] / ((logk[1] - logk[0]) * (logk[1] - logk[2]))
    c2 = logpk[2] / ((logk[2] - logk[0]) * (logk[2] - logk[1]))
    a = c0 + c1 + c2
    logk0 = (c0 * (logk[1] + logk[2]) + c1 * (logk[0] + logk[2]) + c2 * (logk[0] + logk[1])) / (2. * a)
    return float(10. ** logk0)


class TurnOverSpectrum2Template(Spectrum2Template):
    r"""
    Power spectrum template parameterized around the matter turn-over scale.

    The power spectrum shape is modelled as a scale-free parabola in log-log space
    centered on the turn-over wavenumber :math:`k_\mathrm{TO}` (Brieden et al. 2022,
    https://arxiv.org/pdf/2302.07484):

    .. math::
        P(k) = P(k_\mathrm{TO})^{1 - s(k) \cdot x^2}, \quad
        x = \frac{\log_{10}(k)}{\log_{10}(k_\mathrm{TO})} - 1

    where :math:`s = m` for :math:`x > 0` (high-:math:`k` side) and :math:`s = n`
    otherwise.

    AP distortion is parameterized by ``apmode`` (default ``'qap'``).  The
    observables :attr:`DV_times_kTO` and :attr:`DH_over_DM` are set at every call.

    Parameters
    ----------
    k : array, default=None
        Wavenumbers [h/Mpc]. Defaults to ``np.logspace(-3, 1, 400)``.
    z : float, default=1.
        Effective redshift.
    fiducial : str, tuple, dict, or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology used to seed the turn-over scale and growth rate.
    apmode : str, default='qap'
        AP parameterization.  With ``'qap'`` only the anisotropy ratio is free;
        add ``'qisoqap'`` to also free the isotropic dilation.
    eta : float, default=1./3.
        Exponent in :math:`q_\mathrm{iso} = q_\parallel^\eta \, q_\perp^{1-\eta}`.

    Attributes set by ``__call__``
    --------------------------------
    pk_dd, pknow_dd : ndarray
    f, f0, fk : float or ndarray
    qpar, qper : float
        AP distortion ratios, derived from the sampled apmode parameters.
    sigma8, fsigma8, sigma8_fid : float
        sigma8 stays at its fiducial value (no amplitude-rescaling parameter); fsigma8
        tracks the df-scaled growth rate.
    DV_times_kTO, DH_over_DM : float
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

    @classmethod
    def propose_params(cls, apmode='qap'):
        """Return a proposed parameter collection for this template."""
        _prior_pos = dict(limits=[0.5, 1.5])
        _ref_tight = dict(dist='norm', loc=1., scale=0.01)
        return propose_params_multitracer(
            _ap_auto_params(apmode) + [
                Parameter('df', value=1., prior=dict(limits=[0., 2.]),
                          ref=dict(dist='norm', loc=1., scale=0.05), fd_eps=0.02, latex=r'\delta f'),
                Parameter('m', value=0.6, prior=dict(limits=[0., 3.]),
                          ref=dict(dist='norm', loc=0.6, scale=0.1), fd_eps=0.05, latex=r'm'),
                Parameter('n', value=0.9, prior=dict(limits=[0., 3.]),
                          ref=dict(dist='norm', loc=0.9, scale=0.1), fd_eps=0.05, latex=r'n'),
                Parameter('qto', value=1., prior=_prior_pos, ref=_ref_tight, fd_eps=0.005,
                          latex=r'q_\mathrm{TO}'),
                Parameter('dpto', value=1., prior=_prior_pos, ref=_ref_tight, fd_eps=0.005,
                          latex=r'\delta P_\mathrm{TO}'),
            ], tracers=None)

    def __init__(self, k=None, z=1., fiducial='DESI', apmode='qap', eta=1. / 3., params=None):
        vc = type(self).propose_params(apmode=str(apmode))
        if params is not None:
            vc = vc + VariableCollection(params)
        assign_params(self, vc, None)
        # See BAOSpectrum2Template.__init__: _qpar_qper() reads from self.params rather
        # than self.qpar/self.qper, since __call__ reassigns those to the derived output.
        self.params = vc

    def __post_init__(self, k=None, z=1., fiducial='DESI', apmode='qap', eta=1. / 3., params=None):
        from cosmoprimo import constants
        self._apmode = str(apmode)
        self._eta = float(eta)

        if k is None:
            k = np.logspace(-3., 1., 400)
        self.k = np.asarray(k, dtype='f8')
        self.z = float(z)

        self._fiducial = _get_fiducial(fiducial)

        fo = self._fiducial.get_fourier()
        sigma8 = fo.sigma8_z(self.z, of='delta_cb')
        fsigma8 = fo.sigma8_z(self.z, of='theta_cb')
        self._sigma8_fid = float(sigma8)
        self._f_fid = float(fsigma8 / sigma8)

        pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk)
        k0 = 1e-3
        pk1d = pk_interp.to_1d(z=self.z)
        self._f0_fid = float(np.sqrt(fo.pk_interpolator(of='theta_cb', **_kw_pk).to_1d(z=self.z)(k0) / pk1d(k0)))
        self._fk_fid = np.sqrt(fo.pk_interpolator(of='theta_cb', **_kw_pk).to_1d(z=self.z)(self.k) / pk1d(self.k))

        # Turn-over scale from fiducial PK.
        k_grid = pk_interp.k
        pk_grid = pk_interp(k_grid, z=self.z)
        self._kTO_fid = _find_turn_over(k_grid, pk_grid)
        self._pkTO_dd_fid = float(pk1d(self._kTO_fid))

        # Fiducial distance combinations used for observable outputs.
        DH_fid = float(constants.c / 1e3 / (100. * self._fiducial.efunc(self.z)))
        DM_fid = float(self._fiducial.comoving_transverse_distance(self.z))
        DV_fid = DH_fid ** eta * DM_fid ** (1. - eta) * self.z ** (1. / 3.)
        self._DV_times_kTO_fid = DV_fid * self._kTO_fid
        self._DH_over_DM_fid = DH_fid / DM_fid

    def _qpar_qper(self):
        if self._apmode == 'qparqper':
            return self.params['qpar'].value, self.params['qper'].value
        if self._apmode == 'qiso':
            q = self.params['qiso'].value
            return q, q
        if self._apmode == 'qap':
            qap = self.params['qap'].value
            return qap ** (1. - self._eta), qap ** (-self._eta)
        qiso, qap = self.params['qiso'].value, self.params['qap'].value
        return qiso * qap ** (1. - self._eta), qiso * qap ** (-self._eta)

    def ap_k_mu(self, k, mu):
        """Apply AP distortion to a (k, mu) grid; returns (jac, kap, muap)."""
        qpar, qper = self._qpar_qper()
        return _ap_k_mu(k, mu, qpar, qper)

    def __call__(self):
        qto = self.qto.value
        dpto = self.dpto.value
        df = self.df.value
        m = self.m.value
        n = self.n.value
        kTO = self._kTO_fid * qto
        pkTO = self._pkTO_dd_fid * dpto
        x = jnp.log10(self.k) / jnp.log10(kTO) - 1.
        self.pk_dd = jnp.where(x > 0., pkTO ** (1. - m * x ** 2.), pkTO ** (1. - n * x ** 2.))
        self.pknow_dd = self.pk_dd
        self.f = self._f_fid * df
        self.f0 = self._f0_fid * df
        self.fk = self._fk_fid * df
        qpar, qper = self._qpar_qper()
        self.qpar = qpar
        self.qper = qper
        qiso = qpar ** self._eta * qper ** (1. - self._eta)
        self.DV_times_kTO = qiso * self._DV_times_kTO_fid
        self.DH_over_DM = (qpar / qper) * self._DH_over_DM_fid
        # No amplitude-rescaling parameter: sigma8 stays at its fiducial value;
        # fsigma8 tracks the df-scaled growth rate.
        self.sigma8 = jnp.asarray(self._sigma8_fid)
        self.fsigma8 = self.f * self.sigma8
        self.sigma8_fid = jnp.asarray(self._sigma8_fid)
        return self.pk_dd

    def tree_flatten(self):
        return ([self.pk_dd, self.pknow_dd, self.f, self.f0, self.fk, self.qpar, self.qper,
                 self.sigma8, self.fsigma8, self.sigma8_fid], {'k': self.k})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (obj.pk_dd, obj.pknow_dd, obj.f, obj.f0, obj.fk, obj.qpar, obj.qper,
         obj.sigma8, obj.fsigma8, obj.sigma8_fid) = children
        obj.k = aux['k']
        return obj


# ── Direct wiggle-split template ──────────────────────────────────────────────

class DirectWiggleSplitSpectrum2Template(DirectSpectrum2Template):
    r"""
    Direct power spectrum template with independent BAO wiggle rescaling.

    Identical to :class:`DirectSpectrum2Template` but the BAO wiggles are shifted in
    k-space by ``qbao`` (marginalizing over the sound horizon scale) and optionally
    damped by a Gaussian envelope controlled by ``sigmabao`` (Brieden et al. 2021,
    https://arxiv.org/abs/2112.10749).

    The wiggles are computed from the current cosmology as
    ``pk(k / qbao) - pknow(k / qbao)`` using the cosmo requirements registered on
    a fine internal k grid, then damped by :math:`\exp(-(k\,\sigma_\mathrm{BAO})^2)`.

    Parameters
    ----------
    with_now : str, default='peakaverage'
        No-wiggle filter engine.  Unlike the base class, this defaults to
        ``'peakaverage'`` because the wiggle split is always needed.
    (All other parameters as in :class:`DirectSpectrum2Template`.)

    Additional free parameters
    --------------------------
    qbao : float, default=1.
        BAO scale dilation (shifts wiggles in k).  ``qbao = 1`` is no shift.
    sigmabao : float, default=0. (fixed)
        Gaussian damping scale [h/Mpc]\ :sup:`-1`.
    """

    @classmethod
    def install(cls, installer):
        installer.pip('git+https://github.com/cosmodesi/cosmoprimo')

    @classmethod
    def propose_params(cls, engine='class', fiducial=None):
        """Return ``qbao`` and ``sigmabao`` parameters (cosmo params live in the dep)."""
        return VariableCollection([
            Parameter('qbao', value=1., prior=dict(limits=[0.5, 1.5]),
                      ref=dict(dist='norm', loc=1., scale=0.01),
                      fd_eps=0.005, latex=r'q_\mathrm{BAO}'),
            Parameter('sigmabao', value=0., fixed=True, prior=dict(limits=[0., 30.]),
                      ref=dict(dist='norm', loc=0., scale=10.),
                      fd_eps=1., latex=r'\Sigma_\mathrm{BAO}'),
        ])

    def __init__(self, k=None, z=1., fiducial='DESI', engine='class',
                 with_now='peakaverage', only_now=False, cosmo=None):
        super().__init__(k=k, z=z, fiducial=fiducial, engine=engine,
                         with_now=with_now, only_now=only_now, cosmo=cosmo)
        assign_params(self, type(self).propose_params(), None)

    def __post_init__(self, k=None, z=1., fiducial='DESI', engine='class',
                      with_now='peakaverage', only_now=False, cosmo=None):
        super().__post_init__(k=k, z=z, fiducial=fiducial, engine=engine,
                              with_now=with_now, only_now=only_now, cosmo=cosmo)
        self._k_fine = np.logspace(-3., 1., 2000)
        self.cosmo.add_requirements({
            'fourier.pk':     [{'of': 'delta_cb', 'z': self.z, 'k': self._k_fine}],
            'fourier.pk_now': [{'of': 'delta_cb', 'engine': str(with_now), 'z': self.z, 'k': self._k_fine}],
        })

    def __call__(self):
        super().__call__()  # sets pk_dd, pknow_dd, f, sigma8, qpar, qper, etc.
        qbao = self.qbao.value
        sigmabao = self.sigmabao.value
        # Evaluate pk and pknow on the fine grid for shifted-wiggle interpolation.
        pk_fine = self.cosmo.get_fourier().pk(of='delta_cb', z=self.z, k=self._k_fine)
        pknow_fine = self.cosmo.get_fourier().pk_now(of='delta_cb',
                                    engine=self._with_now, z=self.z, k=self._k_fine)
        k_query = jnp.clip(self.k / qbao, self._k_fine[0], self._k_fine[-1])
        wiggles = jnp.interp(jnp.log10(k_query), jnp.log10(self._k_fine), pk_fine - pknow_fine)
        wiggles = wiggles * jnp.exp(-(self.k * sigmabao) ** 2.)
        self.pk_dd = self.pknow_dd + wiggles
        if self._only_now:
            self.pk_dd = self.pknow_dd
        return self.pk_dd


# ── BAO extractors ─────────────────────────────────────────────────────────────

class BAOExtractor(Calculator):
    r"""Extract BAO distance parameters from a cosmology provider.

    At each call, retrieves :math:`E(z) = H(z)/H_0`, the comoving transverse distance
    :math:`D_M(z)`, and the sound horizon :math:`r_d` from the registered cosmology and
    computes the standard BAO observables, plus their ratios relative to a fixed fiducial:

    .. math::

        q_\parallel = \frac{D_H/r_d}{(D_H/r_d)_{\rm fid}}, \quad
        q_\perp    = \frac{D_M/r_d}{(D_M/r_d)_{\rm fid}}, \quad
        q_{\rm iso} = \frac{D_V/r_d}{(D_V/r_d)_{\rm fid}}, \quad
        q_{\rm ap}  = \frac{D_H/D_M}{(D_H/D_M)_{\rm fid}}

    where :math:`D_H = c/H(z)` and :math:`D_V = D_H^\eta D_M^{1-\eta} z^{1/3}`.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.
    eta : float, default=1./3.
        Exponent defining the DV combination.
    fiducial : str or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology used to normalise the AP ratios.
    cosmo : PrimordialCosmology, optional
        Cosmology provider; a :class:`CosmoprimoCosmology` is created if not given.

    Attributes
    ----------
    DH_over_rd, DM_over_rd, DH_over_DM, DV_over_rd : JAX scalar
        Measured distance combinations.
    qpar, qper, qiso, qap : JAX scalar
        AP ratios relative to fiducial.
    """

    def __init__(self, z=1., eta=1./3., fiducial='DESI', cosmo=None):
        if cosmo is None:
            cosmo = CosmoprimoCosmology(fiducial=fiducial)
        self.cosmo = cosmo

    def __post_init__(self, z=1., eta=1./3., fiducial='DESI', cosmo=None):
        from cosmoprimo import constants
        self.cosmo.add_requirements({
            'background.efunc':                        [{'z': float(z)}],
            'background.comoving_transverse_distance': [{'z': float(z)}],
            'thermodynamics.rs_drag':                  None,
        })
        self.z = float(z)
        self._eta = float(eta)
        self._fiducial = _get_fiducial(fiducial)
        rd_fid = self._fiducial.rs_drag
        DH_fid = constants.c / 1e3 / (100. * self._fiducial.efunc(self.z))
        DM_fid = self._fiducial.comoving_transverse_distance(self.z)
        DV_fid = DH_fid ** self._eta * DM_fid ** (1. - self._eta) * self.z ** (1. / 3.)
        self._DH_over_rd_fid = DH_fid / rd_fid
        self._DM_over_rd_fid = DM_fid / rd_fid
        self._DH_over_DM_fid = DH_fid / DM_fid
        self._DV_over_rd_fid = DV_fid / rd_fid

    def __call__(self):
        from cosmoprimo import constants
        efunc = self.cosmo.get_background().efunc(z=self.z)
        DM = self.cosmo.get_background().comoving_transverse_distance(z=self.z)
        rd = self.cosmo.get_thermodynamics().rs_drag
        DH = constants.c / 1e3 / (100. * efunc)
        DV = DH ** self._eta * DM ** (1. - self._eta) * self.z ** (1. / 3.)
        self.DH_over_rd = DH / rd
        self.DM_over_rd = DM / rd
        self.DH_over_DM = DH / DM
        self.DV_over_rd = DV / rd
        self.qpar = self.DH_over_rd / self._DH_over_rd_fid
        self.qper = self.DM_over_rd / self._DM_over_rd_fid
        self.qiso = self.DV_over_rd / self._DV_over_rd_fid
        self.qap  = self.DH_over_DM / self._DH_over_DM_fid
        return self

    def tree_flatten(self):
        return ([self.DH_over_rd, self.DM_over_rd, self.DH_over_DM, self.DV_over_rd,
                 self.qpar, self.qper, self.qiso, self.qap], {'z': self.z})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (obj.DH_over_rd, obj.DM_over_rd, obj.DH_over_DM, obj.DV_over_rd,
         obj.qpar, obj.qper, obj.qiso, obj.qap) = children
        obj.z = aux['z']
        return obj


class BAOPhaseShiftExtractor(BAOExtractor):
    r"""BAO extractor extended with the neutrino-induced BAO phase shift.

    Adds :attr:`N_eff` (effective number of relativistic species from the cosmology) and
    the derived :attr:`baoshift` parameter

    .. math::

        \phi_{\rm BAO} = \frac{N_{\rm eff} \,(N_{\rm eff,fid} + a_\nu)}{N_{\rm eff,fid}\,(N_{\rm eff} + a_\nu)},
        \quad a_\nu = \tfrac{8}{7}\!\left(\tfrac{11}{4}\right)^{4/3}

    following Baumann et al. 2018 (https://arxiv.org/abs/1803.10741).

    Parameters
    ----------
    Same as :class:`BAOExtractor`.

    Attributes
    ----------
    N_eff : JAX scalar
        Effective number of relativistic species from the cosmology.
    baoshift : JAX scalar
        BAO phase-shift amplitude relative to fiducial.
    """

    def __init__(self, z=1., eta=1./3., fiducial='DESI', cosmo=None):
        super().__init__(z=z, eta=eta, fiducial=fiducial, cosmo=cosmo)

    def __post_init__(self, z=1., eta=1./3., fiducial='DESI', cosmo=None):
        super().__post_init__(z=z, eta=eta, fiducial=fiducial, cosmo=cosmo)
        self.cosmo.add_requirements({'params.N_eff': None})
        self._N_eff_fid = float(self._fiducial.N_eff)

    def __call__(self):
        super().__call__()
        a_nu = 8.0 / 7.0 * (11.0 / 4.0) ** (4.0 / 3.0)
        self.N_eff = self.cosmo.get('params.N_eff')
        self.baoshift = (self.N_eff * (self._N_eff_fid + a_nu)) / (self._N_eff_fid * (self.N_eff + a_nu))
        return self

    def tree_flatten(self):
        leaves, aux = super().tree_flatten()
        return leaves + [self.N_eff, self.baoshift], aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (obj.DH_over_rd, obj.DM_over_rd, obj.DH_over_DM, obj.DV_over_rd,
         obj.qpar, obj.qper, obj.qiso, obj.qap,
         obj.N_eff, obj.baoshift) = children
        obj.z = aux['z']
        return obj


class TurnOverExtractor(Calculator):
    r"""Extract turn-over observables from a cosmology provider.

    Evaluates the matter power spectrum on a fine internal k grid, locates the
    turn-over wavenumber :math:`k_{\rm TO}` with ``jnp.argmax``, and computes

    .. math::

        D_V \cdot k_{\rm TO}, \quad D_H / D_M

    together with the dimensionless ratios relative to a fixed fiducial:

    .. math::

        q_{\rm to} = \frac{D_V \cdot k_{\rm TO}}{(D_V \cdot k_{\rm TO})_{\rm fid}}, \quad
        q_{\rm ap} = \frac{D_H/D_M}{(D_H/D_M)_{\rm fid}}

    Gradient information is obtained via finite differences on the cosmology
    (``jnp.argmax`` has zero gradient), which is consistent with the external-code
    use-case these extractors are designed for.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.
    eta : float, default=1./3.
        Exponent defining the DV combination.
    fiducial : str or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology used to compute fiducial distances and kTO.
    cosmo : PrimordialCosmology, optional
        Cosmology provider; a :class:`CosmoprimoCosmology` is created if not given.

    Attributes
    ----------
    kTO, pkTO_dd : JAX scalar
        Turn-over wavenumber and power at the turn-over.
    DH_over_DM, DV_times_kTO : JAX scalar
        Distance combinations.
    qap, qto : JAX scalar
        AP and turn-over ratios relative to fiducial.

    Reference
    ---------
    https://arxiv.org/abs/2302.07484
    """

    def __init__(self, z=1., eta=1./3., fiducial='DESI', cosmo=None):
        if cosmo is None:
            cosmo = CosmoprimoCosmology(fiducial=fiducial)
        self.cosmo = cosmo

    def __post_init__(self, z=1., eta=1./3., fiducial='DESI', cosmo=None):
        from cosmoprimo import constants
        self._k_fine = np.logspace(-3., 0., 2000)
        self.cosmo.add_requirements({
            'fourier.pk':                              [{'of': 'delta_cb', 'z': float(z), 'k': self._k_fine}],
            'background.efunc':                        [{'z': float(z)}],
            'background.comoving_transverse_distance': [{'z': float(z)}],
        })
        self.z = float(z)
        self._eta = float(eta)
        self._fiducial = _get_fiducial(fiducial)
        # Fiducial turn-over from the full interpolation grid.
        fo = self._fiducial.get_fourier()
        pk_interp = fo.pk_interpolator(of='delta_cb', **_kw_pk)
        self._kTO_fid = _find_turn_over(pk_interp.k, pk_interp(pk_interp.k, z=self.z))
        self._pkTO_dd_fid = float(pk_interp.to_1d(z=self.z)(self._kTO_fid))
        # Fiducial distance combinations.
        DH_fid = float(constants.c / 1e3 / (100. * self._fiducial.efunc(self.z)))
        DM_fid = float(self._fiducial.comoving_transverse_distance(self.z))
        DV_fid = DH_fid ** self._eta * DM_fid ** (1. - self._eta) * self.z ** (1. / 3.)
        self._DH_over_DM_fid = DH_fid / DM_fid
        self._DV_times_kTO_fid = DV_fid * self._kTO_fid

    def __call__(self):
        from cosmoprimo import constants
        pk_fine = self.cosmo.get_fourier().pk(of='delta_cb', z=self.z, k=self._k_fine)
        imax = jnp.argmax(pk_fine)
        k_jnp = jnp.asarray(self._k_fine)
        self.kTO = k_jnp[imax]
        self.pkTO_dd = pk_fine[imax]
        efunc = self.cosmo.get_background().efunc(z=self.z)
        DM = self.cosmo.get_background().comoving_transverse_distance(z=self.z)
        DH = constants.c / 1e3 / (100. * efunc)
        DV = DH ** self._eta * DM ** (1. - self._eta) * self.z ** (1. / 3.)
        self.DH_over_DM = DH / DM
        self.DV_times_kTO = DV * self.kTO
        self.qap = self.DH_over_DM / self._DH_over_DM_fid
        self.qto = self.DV_times_kTO / self._DV_times_kTO_fid
        return self

    def tree_flatten(self):
        return ([self.DH_over_DM, self.DV_times_kTO, self.kTO, self.pkTO_dd,
                 self.qap, self.qto], {'z': self.z})

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.DH_over_DM, obj.DV_times_kTO, obj.kTO, obj.pkTO_dd, obj.qap, obj.qto = children
        obj.z = aux['z']
        return obj



class ShapeFitExtractor(BAOExtractor):
    r"""
    Extract ShapeFit parameters from linear power spectrum.

    Inherits from :class:`BAOExtractor` to simultaneously compute standard BAO distance 
    parameters. At each call, retrieves the ShapeFit parameters :math:`n`, :math:`m`, 
    :math:`A_p`, :math:`f\sqrt{A_p}`, and :math:`f\sigma_r`, plus their ratios relative 
    to a fixed fiducial cosmology.

    Parameters
    ----------
    z : float, default=1.
        Effective redshift.

    kp : float, default=0.03
        Pivot point in ShapeFit parameterization [h/Mpc].

    a : float, default=0.6
        Steepness parameter in ShapeFit parameterization.

    eta : float, default=1./3.
        Exponent defining the :math:`D_V` combination: 
        :math:`q_{\rm iso} = q_\parallel^\eta \, q_\perp^{1-\eta}`.

    n_varied : bool, default=False
        Use second order ShapeFit parameter ``n``.
        This choice changes the definition of parameter ``m`` by including the 
        primordial power spectrum slope.

    dfextractor : str, default='Ap'
        Method to compute the growth rate scaling parameter ``df``. 
        Either 'Ap' (using :math:`f\sqrt{A_p}`) or 'fsigmar' (using :math:`f\sigma_r`).

    r : float, default=8.
        Sphere radius [Mpc/h] to estimate the normalization of the linear power 
        spectrum for the :math:`f\sigma_r` computation.

    with_now : str or False, default='peakaverage'
        Engine for the BAO-filtered smooth power spectrum ('peakaverage', 'wallish2018').
        Set to False to skip (uses the full power spectrum instead).

    fiducial : str or cosmoprimo.Cosmology, default='DESI'
        Fiducial cosmology used to normalise the AP ratios and compute fiducial 
        ShapeFit parameters.

    cosmo : PrimordialCosmology, optional
        Cosmology provider; a :class:`CosmoprimoCosmology` is created if not given.

    Attributes
    ----------
    DH_over_rd, DM_over_rd, DH_over_DM, DV_over_rd : JAX scalar
        Measured BAO distance combinations (inherited from BAOExtractor).
    qpar, qper, qiso, qap : JAX scalar
        AP ratios relative to fiducial (inherited from BAOExtractor).
    n, m, Ap, f_sqrt_Ap, f_sigmar : JAX scalar
        Measured absolute ShapeFit parameters.
    dn, dm, dA, df : JAX scalar
        ShapeFit parameters relative to fiducial.

    Reference
    ---------
    https://arxiv.org/abs/2106.07641
    https://arxiv.org/pdf/2212.04522.pdf
    """

    def __init__(self, z=1., kp=0.03, a=0.6, eta=1./3., n_varied=False, 
                 dfextractor='Ap', r=8., with_now='peakaverage', fiducial='DESI', cosmo=None):
        super().__init__(z=z, eta=eta, fiducial=fiducial, cosmo=cosmo)
        self.kp = float(kp)
        self.a = float(a)
        self.n_varied = bool(n_varied)
        if dfextractor not in ['Ap', 'fsigmar', 'f']:
            raise ValueError(f"dfextractor must be one of ['Ap', 'fsigmar', 'f'], found {dfextractor}")
        self.dfextractor = dfextractor
        self.logger.info(f"Using dfectractor = {dfextractor}")
        self.r = float(r)
        self.with_now = with_now

    def __post_init__(self, z=1., kp=0.03, a=0.6, eta=1./3., n_varied=False, 
                      dfextractor='Ap', r=8., with_now='peakaverage', fiducial='DESI', cosmo=None):
        # Initialize BAOExtractor requirements and fiducial distances
        super().__post_init__(z=z, eta=eta, fiducial=fiducial, cosmo=cosmo)
        
        self.kp = float(kp)
        self.a = float(a)
        self.n_varied = bool(n_varied)
        self.dfextractor = dfextractor
        self.r = float(r)
        self.with_now = with_now
        
        # Fixed dense k-grid for interpolation and sigma_r integration
        self._k_fine = jnp.geomspace(1e-4, 1e2, 2000)
        self._dk = 1e-2
        
        # Add ShapeFit-specific requirements to the cosmology provider
        reqs = {
            'params.n_s': None,
            'thermodynamics.rs_drag': None,
            'fourier.sigma8_z': [
                {'of': 'delta_cb', 'z': self.z},
                {'of': 'theta_cb', 'z': self.z},
            ],
        }
        if self.with_now:
            reqs['fourier.pk_now'] = [{'of': 'delta_cb', 'engine': str(self.with_now), 'z': self.z, 'k': self._k_fine}]
        else:
            reqs['fourier.pk'] = [{'of': 'delta_cb', 'z': self.z, 'k': self._k_fine}]
            
        self.cosmo.add_requirements(reqs)
        
        # Compute and cache all fiducial quantities at compile time using the shared helper
        fiducials = _compute_shapefit_fiducials(
            self._fiducial, self.z, self.kp, self.a, self.with_now, 
            r=self.r, n_varied=self.n_varied
        )
        self.n_fid = fiducials['n_fid']
        self.m_fid = fiducials['m_fid']
        self.Ap_fid = fiducials['Ap_fid']
        self.f_sqrt_Ap_fid = fiducials['f_sqrt_Ap_fid']
        self.f_sigmar_fid = fiducials['f_sigmar_fid']
        self.f_fid = fiducials['f_fid']
        self._sigmar_fid = float(fiducials['sigmar_fid'])

    def __call__(self):
        # 1. Compute standard BAO parameters (DH/rd, qpar, qper, etc.)
        super().__call__() 
        
        # 2. Get cosmological quantities from the JAX wrapper
        sigma8 = self.cosmo.get_fourier().sigma8_z(of='delta_cb', z=self.z)
        fsigma8 = self.cosmo.get_fourier().sigma8_z(of='theta_cb', z=self.z)
        self.sigma8 = sigma8
        self.fsigma8 = fsigma8
        self.f = fsigma8 / sigma8
        
        #self.n = self.cosmo.get('n_s')
        self.n = self.cosmo.get('params.n_s')
        
        
        s = self.cosmo.get('thermodynamics.rs_drag') / self._fiducial.rs_drag
        kp = self.kp / s
        
        # 3. Get no-wiggle power spectrum on the fine grid
        if self.with_now:
            pknow_dd_fine = self.cosmo.get_fourier().pk_now(of='delta_cb', engine=self.with_now, z=self.z, k=self._k_fine)
        else:
            pknow_dd_fine = self.cosmo.get_fourier().pk(of='delta_cb', z=self.z, k=self._k_fine)
            
        # 4. Interpolate at the shifted pivot points kp * (1 +/- dk)
        k_query = kp * jnp.array([1. - self._dk, 1., 1. + self._dk])
        log_k_fine = jnp.log(self._k_fine)
        log_pknow_fine = jnp.log(pknow_dd_fine)
        log_k_query = jnp.log(k_query)
        
        log_pknow_query = jnp.interp(log_k_query, log_k_fine, log_pknow_fine)
        pknow_query = jnp.exp(log_pknow_query)
        
        pknow_kp_minus = pknow_query[0]
        pknow_kp = pknow_query[1]
        pknow_kp_plus = pknow_query[2]
        
        slope_pknow = (jnp.log(pknow_kp_plus) - jnp.log(pknow_kp_minus)) / (jnp.log(k_query[2]) - jnp.log(k_query[0]))
        
        if self.n_varied:
            slope_pk_prim = self.n + 1.0
        else:
            slope_pk_prim = 0.0
            
        self.m = slope_pknow - slope_pk_prim
        self.Ap = 1. / s**3 * pknow_kp
        self.f_sqrt_Ap = self.f * self.Ap**0.5
        
        # 5. Compute f_sigmar using the analytic approximation (Eq. A.12)
        dm = self.m - self.m_fid
        dn = self.n - self.n_fid
        dAp = self.Ap / self.Ap_fid
        # Analytic approximation for (sigmar / sigmar_fid)^2
        tanh_arg = self.a * jnp.log(self._fiducial.rs_drag / self.r)
        dsigmar_sq = dAp * jnp.exp((dm + dn) / self.a * jnp.tanh(tanh_arg))
        dsigmar = jnp.sqrt(dsigmar_sq)
        
        self.sigmar = self._sigmar_fid * dsigmar
        self.f_sigmar = self.f * self.sigmar
        
        # 6. Compute relative ShapeFit parameters
        self.dn = dn
        self.dm = dm
        self.dAp = dAp
        
        if self.dfextractor == 'Ap':
            self.df = self.f_sqrt_Ap / self.f_sqrt_Ap_fid / self.dAp**0.5
        elif self.dfextractor == 'f':
            self.df = self.f / self.f_fid
        else:
            # df = (f_sigmar / f_sigmar_fid) / dsigmar -> isolates f / f_fid
            self.df = self.f_sigmar / self.fsigmar_fid / dsigmar
            

        #jax.debug.print("dm = {} df = {} dAp = {} dn = {}", self.dm, self.df, self.dAp, self.dn)
        return self

    def tree_flatten(self):
        # Get BAO leaves and aux data
        leaves, aux = super().tree_flatten()
        
        # Add ShapeFit static parameters to aux
        aux.update({
            'kp': self.kp, 'a': self.a, 'eta': self._eta,
            'n_varied': self.n_varied, 'dfextractor': self.dfextractor,
            'r': self.r, 'with_now': self.with_now, 'k_fine': self._k_fine
        })
        
        # Append ShapeFit dynamic leaves
        return leaves + [self.sigma8, self.fsigma8, self.f, self.n,
                         self.m, self.Ap, self.f_sqrt_Ap, self.f_sigmar,
                         self.dn, self.dm, self.dAp, self.df], aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        # Unflatten BAO attributes first (first 8 children)
        obj = BAOExtractor.tree_unflatten(aux, children[:8])
        
        # Unflatten ShapeFit attributes (next 12 children)
        (obj.sigma8, obj.fsigma8, obj.f, obj.n,
         obj.m, obj.Ap, obj.f_sqrt_Ap, obj.f_sigmar,
         obj.dn, obj.dm, obj.dAp, obj.df) = children[8:]
         
        # Restore static parameters
        obj.kp = aux['kp']
        obj.a = aux['a']
        obj._eta = aux['eta']
        obj.n_varied = aux['n_varied']
        obj.dfextractor = aux['dfextractor']
        obj.r = aux['r']
        obj.with_now = aux['with_now']
        obj._k_fine = aux['k_fine']
        return obj





