import numpy as np

from desilike.likelihoods.base import BaseCalculator


class ClTheory(BaseCalculator):

    def initialize(self, cls=None, lensing=None, non_linear=None, unit=None, cosmo=None, T0=None):
        self.requested_cls = dict(cls or {})
        self.ell_max_lensed_cls, self.ell_max_lens_potential_cls = 0, 0
        for cl, ellmax in self.requested_cls.items():
            if cl in ['tt', 'ee', 'bb', 'te']: self.ell_max_lensed_cls = max(self.ell_max_lensed_cls, ellmax)
            elif cl in ['pp', 'tp', 'ep']: self.ell_max_lens_potential_cls = max(self.ell_max_lens_potential_cls, ellmax)
            elif cl in ['tb', 'eb']: pass
            else: raise ValueError('Unknown Cl {}'.format(cl))
        if lensing is None:
            lensing = bool(self.ell_max_lens_potential_cls)
        ellmax = max(self.ell_max_lensed_cls, self.ell_max_lens_potential_cls)
        if non_linear is None:
            if bool(self.ell_max_lens_potential_cls) or max(ellmax if 'b' in cl.lower() else 0 for cl, ellmax in self.requested_cls.items()) > 50:
                non_linear = 'mead'
            else:
                non_linear = ''
        self.unit = unit
        allowed_units = [None, 'muK']
        if self.unit not in allowed_units:
            raise ValueError('Input unit must be one of {}, found {}'.format(allowed_units, self.unit))
        self.T0 = T0
        if cosmo is None:
            from desilike.theories.primordial_cosmology import Cosmoprimo
            cosmo = Cosmoprimo()
        self.cosmo = cosmo
        self.cosmo.init.update(lensing=self.cosmo.init.get('lensing', False) or lensing,
                               ellmax_cl=max(self.cosmo.init.get('ellmax_cl', 0), ellmax),
                               non_linear=self.cosmo.init.get('non_linear', '') or non_linear)

    def calculate(self):
        self.cls = {}
        T0 = self.T0 if self.T0 is not None else self.cosmo.T0_cmb
        hr = self.cosmo.get_harmonic()
        if self.ell_max_lensed_cls:
            lensed_cl = hr.lensed_cl(ellmax=self.ell_max_lensed_cls)
        if self.ell_max_lens_potential_cls:
            lens_potential_cl = hr.lens_potential_cl(ellmax=self.ell_max_lens_potential_cls)
        for cl, ellmax in self.requested_cls.items():
            if cl in ['tb', 'eb']:
                tmp = np.zeros(ellmax + 1, dtype='f8')
            if 'p' in cl:
                tmp = lens_potential_cl[cl][:ellmax + 1]
            else:
                tmp = lensed_cl[cl][:ellmax + 1]
            if self.unit == 'muK':
                npotential = cl.count('p')
                unit = (T0 * 1e6)**(2 - npotential)
                tmp = tmp * unit
            self.cls[cl] = tmp

    def get(self):
        return self.cls

    def __getstate__(self):
        state = {}
        for name in ['requested_cls', 'unit']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return {**state, **self.cls}

    def __setstate__(self, state):
        state = state.copy()
        self.unit = state.pop('unit')
        self.cls = state