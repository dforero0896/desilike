import numpy as np

from desilike import setup_logging


def test_integ():
    from desilike.theories.galaxy_clustering.base import BaseTheoryPowerSpectrumMultipolesFromWedges
    from desilike.theories.galaxy_clustering import StandardPowerSpectrumTemplate, KaiserTracerPowerSpectrumMultipoles

    topoles = BaseTheoryPowerSpectrumMultipolesFromWedges(mu=8)
    mu, wmu = topoles.mu, topoles.wmu
    assert np.isclose(np.sum(wmu), 1.)
    template = StandardPowerSpectrumTemplate()
    pk_trapz = KaiserTracerPowerSpectrumMultipoles(template=template, mu=100, method='trapz')()
    pk_leggauss = KaiserTracerPowerSpectrumMultipoles(template=template, mu=20, method='leggauss')()
    print(pk_trapz)
    print(pk_leggauss)


def test_bao():

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, ResummedBAOWigglesTracerPowerSpectrumMultipoles
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerCorrelationFunctionMultipoles, ResummedBAOWigglesTracerCorrelationFunctionMultipoles

    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmapar=3.)
    theory = ResummedBAOWigglesTracerPowerSpectrumMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmas=3.)
    theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmapar=3.)
    theory = ResummedBAOWigglesTracerCorrelationFunctionMultipoles()
    print(theory.runtime_info.pipeline.params)
    theory(qpar=1.1, sigmas=3.)

    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerPowerSpectrumMultipoles, ResummedBAOWigglesTracerPowerSpectrumMultipoles
    from desilike.theories.galaxy_clustering import DampedBAOWigglesTracerCorrelationFunctionMultipoles, ResummedBAOWigglesTracerCorrelationFunctionMultipoles

    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles()
    theory(qpar=1.1, sigmapar=3.)

    from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, StandardPowerSpectrumTemplate
    template = BAOPowerSpectrumTemplate(z=0.1, fiducial='DESI', apmode='qiso')
    theory.init.update(template=template)
    theory(df=0.8)

    template = StandardPowerSpectrumTemplate(z=0.1, fiducial='DESI', apmode='qiso', with_now='peakaverage')
    theory.init.update(template=template)
    theory()


def test_full_shape():

    def test_emulator_likelihood(theory, test_likelihood=True):
        print('Emulating', theory)
        if test_likelihood:
            from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, ObservablesCovarianceMatrix
            from desilike.likelihoods import ObservablesGaussianLikelihood
            if 'Power' in theory.__class__.__name__:
                observable = TracerPowerSpectrumMultipolesObservable(klim={0: [0.05, 0.2, 0.01], 2: [0.05, 0.2, 0.01], 4: [0.05, 0.2, 0.01]},
                                                                     data={}, theory=theory)
            else:
                observable = TracerCorrelationFunctionMultipolesObservable(slim={0: [20, 150, 4], 2: [20, 150, 4], 4: [20, 150, 4]},
                                                                           data={}, theory=theory)
            observable()
            cov = np.eye(observable.flatdata.shape[0])
            likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov)
            for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*']):
                param.update(derived='.best')
            likelihood()
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        #theory()
        bak = theory()
        calculator = theory.pt
        emulator = Emulator(calculator, engine=TaylorEmulatorEngine(order=0))
        emulator.set_samples()
        emulator.fit()
        pt = emulator.to_calculator()
        theory.init.update(pt=pt)
        assert np.allclose(theory(), bak)
        if test_likelihood:
            likelihood()

    from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate

    from desilike.theories.galaxy_clustering import KaiserTracerPowerSpectrumMultipoles, KaiserTracerCorrelationFunctionMultipoles
    theory = KaiserTracerPowerSpectrumMultipoles()
    theory(logA=3.04, b1=1.).shape
    theory = KaiserTracerCorrelationFunctionMultipoles()
    theory(logA=3.04, b1=1.).shape

    from desilike.theories.galaxy_clustering import LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles
    theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    #test_emulator_likelihood(theory)
    theory(dm=0.01, b1=1.).shape
    theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(ells=(0, 2), template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator_likelihood(theory)
    theory(dm=0.01, b1=1.).shape
    theory.pt

    from desilike.theories.galaxy_clustering import EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    theory = EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator_likelihood(theory)
    theory(dm=0.01, b1=1.).shape
    theory = EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles(ells=(0, 2), template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator_likelihood(theory)
    theory(dm=0.01, b1=1.).shape
    theory.pt

    from desilike.theories.galaxy_clustering import LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    theory = LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles(template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator_likelihood(theory)
    theory(dm=0.01, b1=1.).shape
    theory = LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles(ells=(0, 2), template=ShapeFitPowerSpectrumTemplate(z=0.5))
    test_emulator_likelihood(theory)
    theory(dm=0.01, b1=1.).shape
    theory.pt

    from desilike.theories.galaxy_clustering import PyBirdTracerPowerSpectrumMultipoles, PyBirdTracerCorrelationFunctionMultipoles

    theory = PyBirdTracerPowerSpectrumMultipoles(eft_basis='westcoast')
    test_emulator_likelihood(theory)
    theory(logA=3.04, b1=1.).shape
    theory = PyBirdTracerCorrelationFunctionMultipoles(eft_basis='westcoast')
    test_emulator_likelihood(theory)  # no P(k) computed
    theory(logA=3.04, b1=1.).shape


def test_png():

    from desilike.theories.galaxy_clustering import PNGTracerPowerSpectrumMultipoles

    theory = PNGTracerPowerSpectrumMultipoles(method='prim')
    params = dict(fnl_loc=100., b1=2.)
    theory2 = PNGTracerPowerSpectrumMultipoles(method='matter')
    assert np.allclose(theory2(**params), theory(**params), rtol=2e-3)
    assert not np.allclose(theory2(fnl_loc=0.), theory(), rtol=2e-3)



if __name__ == '__main__':

    setup_logging()
    #test_integ()
    test_bao()
    #test_full_shape()
    #test_png()