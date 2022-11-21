"""Classes and functions dedicated to handling samples drawn from likelihood."""

import glob

import numpy as np

from ..parameter import Samples
from .chain import Chain
from .profiles import Profiles, ParameterBestFit, ParameterCovariance, ParameterContours
from . import diagnostics, utils
from .utils import BaseClass


__all__ = ['Samples', 'Chain', 'Profiles', 'ParameterBestFit', 'ParameterCovariance', 'ParameterContours', 'diagnostics']


from desilike.io import BaseConfig


def load_source(source, choice=None, cov=None, burnin=None, params=None, default=False, return_type=None):
    if not utils.is_sequence(source): fns = [source]
    else: fns = source

    sources = []
    for fn in fns:
        if isinstance(fn, str):
            sources += [BaseClass.load(ff) for ff in glob.glob(fn)]
        else:
            sources.append(fn)

    if burnin is not None:
        sources = [source.remove_burnin(burnin) if hasattr(source, 'remove_burnin') else source for source in sources]

    if choice is not None or cov is not None:
        if not all(type(source) == type(sources[0]) for source in sources):
            raise ValueError('Sources must be of same type for "choice / cov"')
        source = sources[0].concatenate(sources) if sources[0] is not None else {}

    toret = []
    if choice is not None:
        if not isinstance(choice, dict):
            choice = {}
        if hasattr(source, 'bestfit'):
            source = source.bestfit
        tmp = {}
        if params is not None:
            params_in_source = [param for param in params if param in source]
            if params_in_source:
                tmp = source.choice(params=params_in_source, **choice)
            params_not_in_source = [param for param in params if param not in params_in_source]
            for param in params_not_in_source:
                tmp[str(param)] = (param.value if default is False else default)
            tmp = [tmp[str(param)] for param in params]
        elif source:
            tmp = source.choice(params=source.params(), return_type=return_type, **choice)
        toret.append(tmp)

    if cov is not None:
        if hasattr(source, 'covariance'):
            source = source.covariance
        tmp = None
        if params is not None:
            params_in_source = [param for param in params if param in source]
            if params_in_source:
                cov = source.cov(params=params_in_source, return_type=None)
                params = [cov._params[param] if params in params_in_source else param for param in params]
            params_not_in_source = [param for param in params if param not in params_in_source]
            sizes = [1 if param in params_not_in_source else cov._sizes[params_in_source.index(param)] for param in params]
            tmp = np.zeros((len(sizes),) * 2, dtype='f8')
            cumsizes = np.cumsum([0] + sizes)
            if params_in_source:
                idx = [params.index(param) for param in params_in_source]
                index = np.concatenate([np.arange(cumsizes[ii], cumsizes[ii + 1]) for ii in idx])
                tmp[np.ix_(index, index)] = cov._value
            idx = [params.index(param) for param in params_not_in_source]
            indices = np.concatenate([np.arange(cumsizes[ii], cumsizes[ii + 1]) for ii in idx])
            indices = (indices,) * 2
            if default is False:
                tmp[indices] = [param.proposal**2 if param.proposal is not None else np.nan for param in params_not_in_source]
            else:
                tmp[indices] = default
            source = ParameterCovariance(tmp, params=params, sizes=sizes)
        if source:
            tmp = source.cov(return_type=return_type)
        toret.append(tmp)

    if len(toret) == 0:
        return sources
    if len(toret) == 1:
        return toret[0]
    return tuple(toret)


class SourceConfig(BaseConfig):

    def __init__(self, data=None, **kwargs):
        if not isinstance(data, dict):
            data = {'fn': data}
        super(SourceConfig, self).__init__(data=data, **kwargs)
        fn = self.pop('fn', self.pop('source', None))
        self.source = load_source(fn, **{k: v for k, v in self.items() if k not in ['choice', 'cov']})

    def choice(self, params=None, default=False, return_type='dict', **choice):
        return load_source(self.source, choice={**self.get('choice', {}), **choice}, params=params, default=default, return_type=return_type)

    def cov(self, params=None, default=False, return_type='nparray'):
        return load_source(self.source, cov=True, params=params, default=default, return_type=return_type)