import numpy as np

from desilike import setup_logging


def test_misc():
    from desilike.differentiation import deriv_nd, deriv_grid

    X = np.linspace(0., 1., 11)[..., None]
    Y = np.linspace(0., 1., 11)[..., None]
    center = X[0]
    print(deriv_nd(X, Y, orders=[(0, 1, 2)], center=center))

    deriv = deriv_grid([(np.array([0]), np.array([0]), 0)] * 3)
    deriv2 = set([tuple(d) for d in deriv])
    print(deriv, len(deriv), len(deriv2))

    deriv = deriv_grid([(np.linspace(-1., 1., 3), [1, 0, 1], 2)] * 3)
    deriv2 = set([tuple(d) for d in deriv])
    print(deriv, len(deriv), len(deriv2))

    deriv = deriv_grid([(np.linspace(-1., 1., 3), [1, 0, 1], 2), (np.linspace(-1., 1., 5), [1, 1, 0, 1, 1], 1)])
    deriv2 = set([tuple(d) for d in deriv])
    print(deriv, len(deriv), len(deriv2))

    deriv = deriv_grid([(np.linspace(-1., 1., 3), [1, 0, 1], 2)] * 20)
    deriv2 = set([tuple(d) for d in deriv])
    print(deriv, len(deriv), len(deriv2))


def test_jax():
    import timeit
    import numpy as np
    from desilike.jax import jax
    from desilike.jax import numpy as jnp

    def f(*values):
        return jnp.sum(jnp.array(values))

    jac = jax.jacrev(f, argnums=(1, 2))
    print(jac(1., 1., 3.))

    jac = jax.jacrev(jac, argnums=(0, 1, 2))
    print(jac(1., 1., 3.))

    a = np.arange(10)
    number = 100000
    d = {}
    d['np-sum'] = {'stmt': "np.sum(a)", 'number': number}
    d['jnp-sum'] = {'stmt': "jnp.sum(a)", 'number': number}

    for key, value in d.items():
        dt = timeit.timeit(**value, globals={**globals(), **locals()}) #/ value['number'] * 1e3
        print('{} takes {: .3f} milliseconds'.format(key, dt))


def test_differentiation():

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate

    from desilike import Differentiation
    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=1.4))
    theory.init.params['power'] = {'derived': True}
    #for param in theory.all_params:
    #    if param.basename != 'sn0': param.update(fixed=True)
    theory(sn0=100.)
    diff = Differentiation(theory, method=None, order=2)
    diff()
    print(diff(sn0=50.)['power'])


def test_solve():

    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate, BandVelocityPowerSpectrumTemplate

    theory = KaiserTracerPowerSpectrumMultipoles(template=BandVelocityPowerSpectrumTemplate(z=0.5, kp=np.arange(0.05, 0.2 + 1e-6, 0.005)))
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={},
                                                         theory=theory)
    footprint = BoxFootprint(volume=1e10, nbar=1e-5)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
    from desilike.emulators import Emulator, TaylorEmulatorEngine
    emulator = Emulator(theory, engine=TaylorEmulatorEngine(order=1))
    emulator.set_samples(method='finite')
    emulator.fit()
    observable.init.update(theory=emulator.to_calculator())

    for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'dptt*']):
        param.update(prior=None, derived='.best')
    likelihood()
    from desilike.utils import Monitor
    with Monitor() as mem:
        mem.start()
        for i in range(10): likelihood(b1=1. + i * 0.1)
        mem.stop()
        print(mem.get('time', average=False))

    theory = KaiserTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={},
                                                         theory=theory)
    footprint = BoxFootprint(volume=1e10, nbar=1e-5)
    cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)()
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)

    from desilike.utils import Monitor
    with Monitor() as mem:
        mem.start()
        for i in range(10): likelihood(b1=1. + i * 0.1)
        mem.stop()
        print(mem.get('time', average=False))


def test_solve():

    from desilike.theories.galaxy_clustering import (DampedBAOWigglesTracerPowerSpectrumMultipoles, KaiserTracerPowerSpectrumMultipoles,
                                                     LPTVelocileptorsTracerPowerSpectrumMultipoles, PyBirdTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate)
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, ObservablesCovarianceMatrix, BoxFootprint
    from desilike.likelihoods import ObservablesGaussianLikelihood

    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    for param in theory.params.select(basename=['al*']): param.update(namespace='LRG', derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={},
                                                         theory=theory)
    covariance = ObservablesCovarianceMatrix(observables=observable, footprints=BoxFootprint(volume=1e10, nbar=1e-2))
    observable.init.update(covariance=covariance())
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood()

    template = ShapeFitPowerSpectrumTemplate(z=0.5)
    #theory = KaiserTracerPowerSpectrumMultipoles(template=template)
    #theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    theory = PyBirdTracerPowerSpectrumMultipoles(template=template)
    #for param in theory.params.select(basename=['df', 'dm', 'qpar', 'qper']): param.update(fixed=True)
    for param in theory.params.select(basename=['alpha*', 'sn*', 'ce*']): param.update(namespace='LRG', derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01]},
                                                         data={},
                                                         theory=theory)
    covariance = ObservablesCovarianceMatrix(observables=observable, footprints=BoxFootprint(volume=1e10, nbar=1e-2))
    observable.init.update(covariance=covariance())
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    #for param in likelihood.all_params.select(basename=['df', 'dm', 'qpar', 'qper']): param.update(fixed=True)

    likelihood()


def test_fisher_galaxy():

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood, SumLikelihood
    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerPowerSpectrumMultipoles, DirectPowerSpectrumTemplate

    theory = KaiserTracerPowerSpectrumMultipoles(template=DirectPowerSpectrumTemplate(z=0.5))
    for param in theory.params.select(basename=['alpha*', 'sn*']): param.update(derived='.best')
    observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.18, 0.01]},
                                                         data='_pk/data.npy', covariance='_pk/mock_*.npy', wmatrix='_pk/window.npy',
                                                         theory=theory)
    likelihood = ObservablesGaussianLikelihood(observables=[observable])
    likelihood.all_params['logA'].update(derived='jnp.log(10 *  {A_s})', prior=None)
    likelihood.all_params['A_s'] = {'prior': {'limits': [1.9, 2.2]}, 'ref': {'dist': 'norm', 'loc': 2.083, 'scale': 0.01}}
    for param in likelihood.all_params.select(name=['m_ncdm', 'w0_fld', 'wa_fld', 'Omega_k']):
        param.update(fixed=False)

    #print(likelihood(w0_fld=-1), likelihood(w0_fld=-1.1))
    #print(likelihood(wa_fld=0), likelihood(wa_fld=0.1))
    from desilike import Fisher
    fisher = Fisher(likelihood)
    #like = fisher()
    #print(like.to_stats())
    from desilike import mpi
    fisher.mpicomm = mpi.COMM_SELF
    like = fisher()
    print(like.to_stats())
    like2 = like.clone(params=['a', 'b'])
    assert np.allclose(like2._hessian, 0.)
    like2 = like.clone(params=['a', 'b'], center=np.ones(2), hessian=np.eye(2))
    assert np.allclose(like2._center, np.ones(2))
    assert np.allclose(like2._hessian, np.eye(2))
    like2 += like


def test_fisher_cmb():
    from desilike import Fisher, FisherGaussianLikelihood
    from desilike.likelihoods.cmb import BasePlanck2018GaussianLikelihood, TTTEEEHighlPlanck2018PlikLikelihood, TTHighlPlanck2018PlikLiteLikelihood,\
                                         TTTEEEHighlPlanck2018PlikLiteLikelihood, TTLowlPlanck2018ClikLikelihood,\
                                         EELowlPlanck2018ClikLikelihood, LensingPlanck2018ClikLikelihood
    from desilike.likelihoods import SumLikelihood
    from desilike.theories.primordial_cosmology import Cosmoprimo
    # Now let's turn to Planck (lite) clik likelihoods
    cosmo = Cosmoprimo(fiducial='DESI')
    likelihoods = [Likelihood(cosmo=cosmo) for Likelihood in [TTTEEEHighlPlanck2018PlikLiteLikelihood, TTLowlPlanck2018ClikLikelihood,\
                                                              EELowlPlanck2018ClikLikelihood, LensingPlanck2018ClikLikelihood]]
    likelihood_clik = SumLikelihood(likelihoods=likelihoods)
    for param in likelihood_clik.all_params:
        param.update(fixed=True)
    likelihood_clik.all_params['m_ncdm'].update(fixed=False)
    fisher_clik = Fisher(likelihood_clik)
    # Planck covariance matrix used above should roughly correspond to Fisher at the Planck posterior bestfit
    # at which logA ~= 3.044 (instead of logA = ln(1e10 2.0830e-9) = 3.036 assumed in the DESI fiducial cosmology)
    fisher_clik = fisher_clik()
    print(fisher_clik.to_stats(tablefmt='pretty'))
    for likelihood in likelihood_clik.likelihoods:
        print(likelihood, likelihood.loglikelihood)
    fisher_likelihood_clik = fisher_clik.to_likelihood()
    print(fisher_likelihood_clik.all_params)
    print(likelihood_clik(), fisher_likelihood_clik())
    fn = '_tests/test.npy'
    fisher_likelihood_clik.save(fn)
    fisher_likelihood_clik2 = FisherGaussianLikelihood.load(fn)
    assert np.allclose(fisher_likelihood_clik2(), fisher_likelihood_clik())


def test_speed():

    import time
    from cosmoprimo.fiducial import DESI
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles, FlexibleBAOWigglesTracerPowerSpectrumMultipoles, FlexibleBAOWigglesTracerCorrelationFunctionMultipoles
    from desilike.theories.galaxy_clustering import FOLPSTracerPowerSpectrumMultipoles, FOLPSTracerCorrelationFunctionMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, BoxFootprint, ObservablesCovarianceMatrix
    from desilike.likelihoods import ObservablesGaussianLikelihood

    footprint = BoxFootprint(volume=1e10, nbar=1e-4)

    for theory_name in ['FOLPS', 'DampedBAOWiggles', 'FlexibleBAOWiggles'][:1]:
        for observable_name in ['power', 'correlation']:
            if observable_name == 'power':
                theory = locals()[theory_name + 'TracerPowerSpectrumMultipoles']()
                observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.02, 0.3, 0.005], 2: [0.02, 0.3, 0.005]},
                                                                     data={},
                                                                     shotnoise=2e4,
                                                                     theory=theory)
            else:
                theory = locals()[theory_name + 'TracerCorrelationFunctionMultipoles']()
                observable = TracerCorrelationFunctionMultipolesObservable(slim={0: [20., 180., 4.], 2: [20., 180., 4.]},
                                                                           data={},
                                                                           theory=theory)

            cov = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=3)()
            observable.init.update(covariance=cov)
            likelihood = ObservablesGaussianLikelihood(observables=[observable])
            likelihood()
            for param in likelihood.all_params.select(basename=theory.template.init.params.basenames()):
                param.update(fixed=True)
            for iparam, param in enumerate(likelihood.all_params.select(basename=['al*_*', 'ml*_*', 'alpha*', 'sn*'])):
                param.update(derived='.best')
            rng = np.random.RandomState(seed=42)
            for i in range(2):
                params = {param.name: param.prior.sample(random_state=rng) for param in likelihood.varied_params}
                likelihood(**params)
            niterations = 10
            t0 = time.time()
            for i in range(niterations):
                params = {param.name: param.prior.sample(random_state=rng) for param in likelihood.varied_params}
                likelihood(**params)
            print(theory_name, observable_name, (time.time() - t0) / niterations)




from desilike.base import BaseCalculator
from desilike.likelihoods import BaseGaussianLikelihood


class AffineModel(BaseCalculator):  # all calculators should inherit from BaseCalculator

    # Model parameters; those can also be declared in a yaml file
    _params = {'a': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}},
               'b': {'value': 0., 'prior': {'dist': 'norm', 'loc': 0., 'scale': 10.}}}

    def initialize(self, x=None):
        # Actual, non-trivial initialization must happen in initialize(); this is to be able to do AffineModel(x=...)
        # without doing any actual work
        self.x = x

    def calculate(self, a=0., b=0.):
        self.y = a * b * self.x + b  # simple, affine model

    # Not mandatory, this is to return something in particular after calculate (else this will just be the instance)
    def get(self):
        return self.y

    # This is only needed for emulation
    def __getstate__(self):
        return {'x': self.x, 'y': self.y}  # dictionary of Python base types and numpy arrays


class Likelihood(BaseGaussianLikelihood):

    def initialize(self, theory=None):
        # Let us generate some fake data
        self.xdata = np.linspace(0., 1., 10)
        mean = np.zeros_like(self.xdata)
        self.covariance = np.eye(len(self.xdata))
        rng = np.random.RandomState(seed=42)
        y = rng.multivariate_normal(mean, self.covariance)
        super(Likelihood, self).initialize(y, covariance=self.covariance)
        # Requirements
        # AffineModel will be instantied with AffineModel(x=self.xdata)
        if theory is None:
            theory = AffineModel()
        self.theory = theory
        self.theory.init.update(x=self.xdata)  # we set x-coordinates, they will be passed to AffineModel's initialize

    @property
    def flattheory(self):
        # Requirements (theory, requested in __init__) are accessed through .name
        # The pipeline will make sure theory.run(a=..., b=...) has been called
        return self.theory.y  # data - model


def test_autodiff():
    import jax
    import jax.numpy as jnp
    from jax import custom_jvp
    """
    from desilike import mpi
    mpicomm = mpi.COMM_WORLD

    def fun(a):
        return mpicomm.bcast(a, root=0)

    jac = jax.jacfwd(fun)
    print(jac(jnp.array(10.)))
    exit()
    """
    """
    def g(x, y):
        return jnp.sin(x) * y

    @custom_jvp
    def f(x, y):
        return g(x, y)

    @f.defjvp
    def f_jvp(primals, tangents):
        return jax.jvp(g, primals, tangents)

    print(f(2., 3.))
    y, y_dot = jax.jvp(f, (2., 3.), (1., 0.))
    print(y)
    print(y_dot)
    print(jax.jacfwd(f)(2., 3.))


    @custom_jvp
    def f(x, y):
        return jnp.sin(x) * y

    @f.defjvp
    def f_jvp(primals, tangents):
        x, y = primals
        x_dot, y_dot = tangents
        primal_out = f(x, y)
        tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
        return primal_out, tangent_out

    print(f(2., 3.))
    y, y_dot = jax.jvp(f, (2., 3.), (1., 0.))
    print(y)
    print(y_dot)
    print(jax.jacfwd(f)(2., 3.))
    """

    """
    likelihood = Likelihood()

    fun = likelihood

    fun = jax.jit(jax.vmap(fun))
    print(fun({'a': jnp.ones(3), 'b': jnp.ones(3)}))
    """
    likelihood = Likelihood()
    """
    from desilike import Differentiation

    likelihood = Likelihood()
    theory = likelihood.theory
    theory.init.params['y'] = {'derived': True}

    diff = Differentiation(theory, method=None)
    print(diff(b=0.)['y'])
    print(diff(b=1.)['y'])
    """
    likelihood.all_params['a'].update(derived='.marg')
    likelihood()
    fun = likelihood

    def logdensity_fn(b):
        return likelihood(b=b)

    #likelihood(b=0.02)
    grad = jax.value_and_grad(logdensity_fn, argnums=0)
    grad(0.01)
    grad(0.04)



if __name__ == '__main__':

    setup_logging()
    #test_misc()
    #test_differentiation()
    #test_solve()
    test_fisher_galaxy()
    #test_fisher_cmb()
    #test_speed()
    #test_jax()
    #test_autodiff()
