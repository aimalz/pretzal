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
import h5py
import emcee
import matplotlib.pyplot as plt
import corner
import logging

import sed_model_galsim as sedmod

plt.style.use('ggplot')

# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class PhotometryData(object):
    """
    A likelihood model for multi-band photometry observations
    """
    def __init__(self):
        self.model = sedmod.SEDModelGalSim()
        self.use_prior = True

    def load(self, infile, source_index=0):
        """
        Load measured magnitudes from file

        @param infile       Name of the data file to load
        @param source_index Index of the source colors to select
        """
        # FIXME: Replace hard-coded values with those from input file
        self.filter_names = ['u', 'g', 'r', 'i', 'z', 'y']
        # self.data = np.array([22., 22.1, 22.2, 22.3, 22.4, 22.5])
        self.sigma_sq = 0.01 ## Hard-coded mag rms of 0.1

        dat = np.loadtxt(infile)

        self.data = dat[source_index, :]
        return None

    def lnprior(self, p):
        valid_params = self.model.set_params(p)
        try:
            r = self.model.get_magnitude('r')
        except ValueError:
            return -np.inf
        lnp_mag = -0.5 * (r - 20.)**2 / 24.
        lnp_z = -(self.model.redshift / 3.)**2
        return lnp_mag + lnp_z

    def lnlike(self, p, *args, **kwargs):
        valid_params = self.model.set_params(p)
        if valid_params:
            # m = np.array([self.model.get_magnitude(f) for f in self.filter_names])
            try:
                ## If the SED is redshifted out of nominal wavelength range,
                ## then we can get a ValueError from GalSim. This is a bad 
                ## model parameter choice and should return a small likelihood.
                m = self.model.get_colors()
            except ValueError:
                return -np.inf
            delta = self.data - m
            chisq = np.sum(delta**2 / self.sigma_sq)
            return -0.5 * chisq
        else:
            return -np.inf

    def __call__(self, p, *args, **kwargs):
        lnp = self.lnlike(p, *args, **kwargs)
        if self.use_prior:
            lnp += self.lnprior(p)
        return lnp


class PhotometryPrior(object):

    def __init__(self):
        self.model = sedmod.SEDModelGalSim()

    def lnprior(self, p):
        valid_params = self.model.set_params(p)
        try:
            r = self.model.get_magnitude('r')
        except ValueError:
            return -np.inf
        lnp_mag = -0.5 * (r - 20.)**2 / 24.
        lnp_z = -(self.model.redshift / 3.)**2
        return lnp_mag + lnp_z

    def __call__(self, p, *args, **kwargs):
        return self.lnprior(p)


def do_sampling(args, phot, sampler_type="ensemble"):
    """
    @brief      Run MCMC 
    
    @param      args  Command line arguments passed from main()
    @param      phot  An instance of PhotometryData class
    
    @return     MCMC parameter samples and ln-posteriors
    """
    p0 = phot.model.get_params()
    print "Starting params:", p0
    nvars = len(p0)
    p0 = emcee.utils.sample_ball(p0, np.ones_like(p0) * 0.1, args.nwalkers)

    if sampler_type == "ensemble":
        sampler = emcee.EnsembleSampler(args.nwalkers,
                                        nvars,
                                        phot,
                                        threads=args.nthreads)
    elif sampler_type == "parallel":
        phot_prior = PhotometryPrior()
        phot.use_prior = False
        sampler = emcee.PTSampler(args.ntemps,
                                  args.nwalkers,
                                  nvars, 
                                  phot,
                                  phot_prior)
    else:
        raise KeyError("Unsupported sampler type")

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
    return np.array(pps), np.array(lnps)


def write_results(args, pps, lnps):
    logging.info("Writing MCMC results to {}".format(args.outfile))
    f = h5py.File(args.outfile, 'w')
    if "post" in f:
        del f["post"]
    post = f.create_dataset("post", data=np.transpose(np.dstack(pps), [2,0,1]))
    if "logprobs" in f:
        del f["logprobs"]
    logprobs = f.create_dataset("logprobs", data=np.vstack(lnps))
    f.close()
    return None


def plot(args, pps, lnps, plotfile, keeplast=0):
    n = len(sedmod.k_SED_names)
    paramnames = ['z'] + ['sed_mag{:d}'.format(i+1) for i in xrange(n)]
    print "paramnames:", paramnames
    print "data:", np.vstack(pps).shape

    # truths = np.loadtxt("sedz_test_truths.txt")
    truths = [1.12, 25.9, 25.9, 25.9, 25.9]

    fig = corner.corner(np.vstack(pps[-keeplast:,:, 0:(n+1)]),
                          labels=paramnames, 
                          truths=truths)
    logging.info("Saving {}".format(plotfile))
    fig.savefig(plotfile)

    ## Walkers plot

    # First determine size of plot
    nparams = len(paramnames) + 1  # +1 for lnprob plot
    # Try to make plot aspect ratio near golden
    ncols = int(np.ceil(np.sqrt(nparams*1.618)))
    nrows = int(np.ceil(1.0*nparams/ncols))

    fig = plt.figure(figsize = (3.0*ncols,3.0*nrows))

    for i, p in enumerate(paramnames):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.plot(pps[:, :, i])
        ax.set_ylabel(p)
    ax = fig.add_subplot(nrows, ncols, i+2)
    ax.plot(lnps)
    ax.set_ylabel('ln(prob)')
    fig.tight_layout()
    outfile = "walkers.png"
    logging.info("Saving {}".format(outfile))
    fig.savefig(outfile)
    return None


# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Draw samples of (z, SED amplitude) parameters via MCMC.')

    parser.add_argument("--outfile", type=str, default="sedz_out.h5",
                        help="Name of the file for MCMC output (Default: sedz_out.h5)")

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

    parser.add_argument("--sampler_type", type=str, default="ensemble",
                        help="Type of emcee sampler ['ensemble', 'parallel']")

    parser.add_argument("--ntemps", type=int, default=20,
                        help="Number of temperatures for Parallel Tempering (default: 20)")

    args = parser.parse_args()
    logging.debug('--- Starting MCMC sampling')

    phot = PhotometryData()
    infile = "../dat/X_test.dat"
    # infile = "sedz_test.dat"
    phot.load(infile, 1)

    pps, lnps = do_sampling(args, phot, sampler_type=args.sampler_type)

    write_results(args, pps, lnps)
    print "pps:", pps.shape
    plot(args, pps, lnps, plotfile="corner.png")

    logging.debug('--- Sampler finished')
    return 0

if __name__ == "__main__":
    sys.exit(main())
