"""Learned densities over parameters -- the non-Gaussian replacement for a covariance.

    density = fit_density(chain, ['qpar', 'qper', 'df', 'dm', 'dA'], kind='gmm')
    density.write('shapefit_density.h5')
    likelihood = ObservablesDensityLikelihood(observables=[observable], density=density)
"""

import jax

# desilike.base sets this at import, but a density can be fitted and evaluated without going
# through it (a script that only reads a chain and writes a cached density).  A float32 density
# inside a float64 pipeline loses ~7 digits in exactly the term that matters -- the log-determinant
# -- and does so silently, so the subpackage states the precision it requires.
jax.config.update('jax_enable_x64', True)

from .base import Density, fit_density, chain_to_array, prepare_training_samples
from .gmm import GaussianDensity, GaussianMixtureDensity


def __getattr__(name):
    """Resolve the flow backend on first use (PEP 562): importing a density must not import
    flowjax, which is an optional extra (and Python >= 3.11 only)."""
    if name == 'FlowDensity':
        from .flow import FlowDensity
        globals()[name] = FlowDensity
        return FlowDensity
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(globals()) + ['FlowDensity']
