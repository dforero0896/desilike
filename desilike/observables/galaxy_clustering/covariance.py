import numpy as np
from scipy import special

from desilike.theories.primordial_cosmology import get_cosmo
from desilike.base import BaseCalculator, EnsembleCalculator
from desilike.utils import BaseClass
from desilike import utils
from .power_spectrum import TracerPowerSpectrumMultipolesObservable
from .correlation_function import TracerCorrelationFunctionMultipolesObservable


def integral_legendre_product(ells, range=(-1, 1), norm=False):
    r"""
    Return integral of product of Legendre polynomials.

    Parameters
    ----------
    ells : int, list, tuple
        Order(s) of Legendre polynomials to multiply together.
    range : tuple, default=(-1,1)
        :math:`\mu`-integration range.
    norm : bool, default=False
        Whether to normalize integral by the :math:`\mu`-integration range.

    Returns
    -------
    toret : float
        Integral of product of Legendre polynomials.
    """
    poly = special.legendre(0)  # 1
    if np.ndim(ells) == 0: ells = [ells]
    for ell in ells:
        poly *= special.legendre(ell)
    integ = poly.integ()
    toret = integ(range[-1]) - integ(range[0])
    if norm:
        toret /= (range[-1] - range[0])
    return toret


def _interval_intersection(*intervals):
    # Return intersection of input intervals
    return (max(interval[0] for interval in intervals), min(interval[1] for interval in intervals))


def _interval_empty(interval):
    # Return whether input interval is empty, i.e. upper bound lower than lower bound
    return interval[0] >= interval[1]


class BaseFootprint(BaseClass):

    def __init__(self, nbar=None, size=None, volume=None):
        if nbar is None and size is None:
            raise ValueError('provide either "size" (number of objects) or "nbar" (mean comoving density in (Mpc/h)^(-3))')
        for name in ['nbar', 'size', 'volume']:
            value = locals()[name]
            setattr(self, '_' + name, None if value is None else np.asarray(value))

    @property
    def volume(self):
        return self._volume

    @property
    def size(self):
        if self._size is not None:
            return self._size
        try:
            area = self.area
        except AttributeError:
            return self._nbar * self.volume

    @property
    def shotnoise(self):
        return self.volume / self.size

    def __and__(self, other):
        return self.__class__(nbar=self._nbar + other._nbar, volume=min(self.volume, other.volume))


class BoxFootprint(BaseFootprint):

    pass


class CutskyFootprint(BaseFootprint):

    def __init__(self, nbar=None, size=None, area=None, zrange=None, cosmo=None):
        if nbar is None and size is None:
            raise ValueError('provide either "size" (number of objects) or "nbar" (angular density in (deg)^(-2))')
        if area is None or zrange is None:
            raise ValueError('provide area (in deg^2) and zrange (zmin, zmax)')
        for name in ['area', 'zrange', 'nbar']:
            value = np.asarray(locals()[name])
            if value.size <= 1: value.shape = ()
            setattr(self, '_' + name, value)
        self._size = size
        argsort = np.argsort(self._zrange)
        self._zrange = self._zrange[argsort]
        if self._nbar.size == self._zrange.size: self._nbar = self._nbar[argsort]
        self.cosmo = cosmo

    @property
    def cosmo(self):
        if self._cosmo is None:
            raise ValueError('Provide cosmology')
        return self._cosmo

    @cosmo.setter
    def cosmo(self, cosmo):
        self._cosmo = get_cosmo(cosmo)

    @property
    def volume(self):
        volume = self.cosmo.comoving_radial_distance(self._zrange)**3
        return self.area / (180. / np.pi)**2 / 3. * np.diff(volume).sum()

    @property
    def area(self):
        if self._area.ndim == 0:
            return self._area
        return np.mean(self._area) * (180. / np.pi)**2 * (4. * np.pi)

    @property
    def zavg(self):
        z = (self._zrange[:-1] + self._zrange[1:]) / 2.
        if self._nbar.ndim:
            volume = np.diff(self.cosmo.comoving_radial_distance(self._zrange)**3)
            nbar = (self._nbar[:-1] + self._nbar[1:]) / 2.
            return np.average(z, weights=nbar * volume)
        return np.mean(z)

    @property
    def size(self):
        if self._size is not None:
            return self._size
        if self._nbar.ndim:
            volume = np.diff(self.cosmo.comoving_radial_distance(self._zrange)**3)
            nbar = (self._nbar[:-1] + self._nbar[1:]) / 2.
            return self.area / (180. / np.pi)**2 * np.sum(nbar * volume)
        return self.area * self._nbar

    def __and__(self, other):
        if self._area.ndim == 0 or other._area.ndim == 0:
            area = min(self.area, other.area)
        else:
            area = self._area * other._area
        zrange = np.unique(np.concatenate([self._zrange, other._zrange], axis=0))
        mask = (zrange >= max(self._zrange[0], other._zrange[0])) & (zrange <= min(self._zrange[-1], other._zrange[-1]))
        zrange = zrange[mask]
        if self._size is not None or other._size is not None or self._nbar.ndim == 0 or other._nbar.ndim == 0:
            nbar = self._nbar + other._nbar
        else:
            nbar = np.interp(zrange, self._zrange, self._nbar) + np.interp(zrange, other._zrange, other._nbar)
        if zrange.size < 2:
            zrange = [zrange[0], zrange[1]]
            nbar *= 0.
        return self.__class__(nbar=nbar, zrange=zrange, area=area, cosmo=self.cosmo)


class ObservablesCovarianceMatrix(BaseClass):

    """Warning: does not handle cross-correlations of different tracers!"""

    def __init__(self, observables, footprints=None, theories=None, resolution=1):
        if not utils.is_sequence(observables):
            observables = [observables]
        self.observables = EnsembleCalculator(calculators=observables).runtime_info.initialize()
        if not utils.is_sequence(footprints):
            footprints = [footprints] * len(self.observables)
        self.footprints = []
        for footprint, observable in zip(footprints, self.observables):
            if footprint is None: footprint = observable.footprint
            self.footprints.append(footprint.copy())
        if theories is None: theories = [None] * len(self.observables)
        self.theories = theories
        self.resolution = int(resolution)
        if self.resolution <= 0:
            raise ValueError('resolution must be a strictly positive integer')

    def __call__(self, **params):
        self.run(**params)
        return self.covariance

    def run(self, **params):
        self.observables(**params)
        self.cosmo = None
        if any(isinstance(footprint, CutskyFootprint) for footprint in self.footprints):
            for footprint in self.footprints:
                self.cosmo = getattr(footprint, 'cosmo', None)
                if self.cosmo is not None: break
            if self.cosmo is None:
                for observable in self.observables:
                    for calculator in observable.runtime_info.pipeline.calculators:
                        if isinstance(calculator, Cosmoprimo):
                            self.cosmo = calculator
                            break
        for footprint in self.footprints: footprint.cosmo = self.cosmo
        covariance = [[None for o in self.observables] for o in self.observables]
        for io1, o1 in enumerate(self.observables):
            for io2, o2 in enumerate(self.observables[:io1 + 1]):
                covariance[io1][io2] = c = self._run(io1, io2)
                if io2 == io1:
                    covariance[io2][io1] = (c + c.T) / 2.  # just for numerical accuracy
                else:
                    covariance[io2][io1] = c.T
        self.covariance = np.bmat(covariance).A

    def _run(self, io1, io2):
        auto = io2 == io1
        ios = [io1, io2]
        volume = (self.footprints[io1] & self.footprints[io2]).volume
        cache = {}

        def get_pk(observable, footprint, theory=None):
            if theory is None:
                for calculator in observable.runtime_info.pipeline.calculators[::-1]:
                    if hasattr(calculator, 'k') and hasattr(calculator, 'power') and not isinstance(calculator.power, BaseCalculator) and np.ndim(calculator.k) == 1:
                        theory = calculator
                        break
            if theory is None:
                raise ValueError('Theory must be provided for observable {}'.format(observable))

            def pk(k, ell=0):
                ill = theory.ells.index(ell)
                return np.interp(k, theory.k, theory.power[ill] + (ell == 0) * footprint.shotnoise).reshape(k.shape)

            pk.ells = theory.ells
            pk.k = theory.k

            return pk

        def get_sigma_k(pk1, pk2, ell1, ell2, k):
            prefactor = (2 * ell1 + 1) * (2 * ell2 + 1) / volume
            toret = 0.
            ells = [ell1, ell2]
            for ell1 in pk1.ells:
                for ell2 in pk2.ells:
                    toret += pk1(k=k, ell=ell1) * pk2(k=k, ell=ell2) * integral_legendre_product([ell1, ell2] + ells, range=(-1, 1))
            return prefactor * toret

        def get_bin_volume(bin, edges=True):
            bin = np.asarray(bin)
            if edges:
                return 4. / 3. * np.pi * (bin[1:]**3 - bin[:-1]**3)
            return 4. * np.pi * bin**2 * utils.weights_trapz(bin)

        pks = [get_pk(self.observables[io], self.footprints[io], theory=self.theories[io]) for io in ios]

        def get_integ_points(bin):
            return np.linspace(*bin, self.resolution + 2)[1:-1]

        obs = [self.observables[io] for io in ios]

        if all(isinstance(o, TracerPowerSpectrumMultipolesObservable) for o in obs):

            def get_bin_cov(obs, ells, ibins):
                ills = [o.ells.index(ell) for o, ell in zip(obs, ells)]
                bins = [o.kedges[ill][ibin:ibin + 2] for o, ill, ibin in zip(obs, ills, ibins)]
                bin = _interval_intersection(*bins)
                if _interval_empty(bin):
                    return 0.
                k = get_integ_points(bin)
                return (2. * np.pi)**3 * np.mean(get_bin_volume(bin) / np.prod([get_bin_volume(bin) for bin in bins]) * get_sigma_k(*pks, *ells, k))

        if isinstance(obs[0], TracerCorrelationFunctionMultipolesObservable) and isinstance(obs[1], TracerPowerSpectrumMultipolesObservable):

            def get_bin_cov(obs, ells, ibin):
                ills = [o.ells.index(ell) for o, ell in zip(obs, ells)]
                bins = [edges[ill][i:i + 2] for edges, ill, i in zip([obs[0].sedges, obs[1].kedges], ills, ibin)]
                s, k = [get_integ_points(bin) for bin in bins]
                weights = np.sum(s[:, None]**2 * special.spherical_jn(ells[0], s[:, None] * k), axis=0) / np.sum(s**2, axis=0)
                sigmak = np.mean(get_sigma_k(*pks, *ells, k), axis=-1)
                return np.sign(1j ** ells[0]).real * np.sum(sigmak * weights)

        if isinstance(obs[1], TracerCorrelationFunctionMultipolesObservable) and isinstance(obs[0], TracerPowerSpectrumMultipolesObservable):
            return self._run(io2, io1).T

        if all(isinstance(o, TracerCorrelationFunctionMultipolesObservable) for o in obs):

            def get_bin_cov(obs, ells, ibin):
                ills = [o.ells.index(ell) for o, ell in zip(obs, ells)]
                bins = [o.sedges[ill][i:i + 2] for o, ill, i in zip(obs, ills, ibin)]
                if 'k' in cache:
                    k = cache['k']
                else:
                    ks = [pk.k for pk in pks]
                    k = np.unique(np.concatenate(ks, axis=0))
                    k = cache['k'] = k[(k >= max(min(k) for k in ks)) & (k <= min(max(k) for k in ks))]
                if ells in cache:
                    sigmak = cache[ells]
                else:
                    sigmak = cache[ells] = get_sigma_k(*pks, *ells, k) * get_bin_volume(k, edges=False)
                ss = [get_integ_points(bin) for bin in bins]
                weights = np.prod([np.sum(s[:, None]**2 * special.spherical_jn(ell, s[:, None] * k), axis=0) / np.sum(s**2, axis=0) for s, ell in zip(ss, ells)], axis=0)
                return np.sign(1j ** sum(ells)).real / (2. * np.pi)**3 * np.sum(sigmak * weights)

        covariance = []
        for ill1, ell1 in enumerate(obs[0].ells):
            row = []
            for ill2, ell2 in enumerate(obs[1].ells):
                n1, n2 = [len(o.data[ill]) for o, ill in zip(obs, [ill1, ill2])]
                row.append(np.array([[get_bin_cov(obs, (ell1, ell2), (i1, i2)) for i2 in range(n2)] for i1 in range(n1)], dtype='f8'))
            covariance.append(row)

        return np.bmat(covariance).A