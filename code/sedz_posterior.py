#!/usr/bin/env python
# encoding: utf-8
"""
sedz_posterior.py

Draw samples from the joint posterior of redshift and SED template amplitudes
given multi-band photometry measurements.
"""
import argparse
import sys
import math
import os.path
import numpy as np
import emcee
import logging

import sed_model_galsim as sedmod

# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class PhotometryData(object):
    """
    A likelihood model for multi-band photometry observations
    """
    def __init__(self):
        self.model = sedmod.SEDModelGalSim()

    def load(self, infile):
        """
        Load measured magnitudes from file
        """
        # FIXME: Replace hard-coded values with those from input file
        self.filter_names = ['u', 'g', 'r', 'i', 'z', 'y']
        self.data = np.array([20., 20., 20., 20., 20., 20.])
        self.sigma_sq = 0.25
        return None

    def lnprior(self, p):
        return 0.0

    def lnlike(self, p, *args, **kwargs):
        self.model.set_params(p)
        m = np.array([self.model.get_magnitude(f) for f in self.filter_names])
        delta = self.data - m
        chisq = np.sum(delta**2 / self.sigma_sq)
        return -0.5 * chisq

    def __call__(self, p, *args, **kwargs):
        return self.lnlike(p, *args, **kwargs) + self.lnprior(p)        


def do_sampling(args, phot):
    """
    @brief      Run MCMC 
    
    @param      args  Command line arguments passed from main()
    @param      phot  An instance of PhotometryData class
    
    @return     MCMC parameter samples and ln-posteriors
    """
    p0 = phot.model.get_params()
    nvars = len(p0)
    p0 = emcee.utils.sample_ball(p0, np.ones_like(p0) * 0.01, args.nwalkers)
    sampler = emcee.EnsembleSampler(args.nwalkers,
                                    nvars,
                                    phot,
                                    threads=args.nthreads)
    nburn = max([1,args.nburn])
    logging.info("Burning with {:d} steps".format(nburn))
    pp, lnp, rstate = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    pps = []
    lnps = []
    lnpriors = []
    logging.info("Sampling")
    for i in range(args.nsamples):
        if np.mod(i+1, 10) == 0:
            print "\tStep {:d} / {:d}, lnp: {:5.4g}".format(i+1, args.nsamples,
                np.mean(pp))
        pp, lnp, rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
        if not args.quiet:
            print i, np.mean(lnp)
            print np.mean(pp, axis=0)
            print np.std(pp, axis=0)
        lnprior = np.array([phot.lnprior(p) for p in pp])
        pps.append(np.column_stack((pp.copy(), lnprior)))
        lnps.append(lnp.copy())
    return pps, lnps

# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Draw samples of (z, SED amplitude) parameters via MCMC.')

    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for pseudo-random number generator")

    parser.add_argument("--nsamples", default=100, type=int,
                        help="Number of samples for each emcee walker "+
                             "(Default: 100)")

    parser.add_argument("--nwalkers", default=32, type=int,
                        help="Number of emcee walkers (Default: 16)")

    parser.add_argument("--nburn", default=50, type=int,
                        help="Number of burn-in steps (Default: 50)")

    parser.add_argument("--nthreads", default=1, type=int,
                        help="Number of threads to use (Default: 1)")

    parser.add_argument("--quiet", action='store_true')

    args = parser.parse_args()
    logging.debug('--- Starting MCMC sampling')

    phot = PhotometryData()
    phot.load("")

    pps, lnps = do_sampling(args, phot)

    print pps

    logging.debug('--- Sampler finished')
    return 0

if __name__ == "__main__":
    sys.exit(main())
