#!/usr/bin/env python
# encoding: utf-8
"""
sedz_posterior.py

Draw samples from the joint posterior of redshift and SED template amplitudes
given multi-band photometry measurements.
"""
import argparse
import sys
import os.path
import numpy as np
import emcee


class PhotometryData(object):
	"""
	A likelihood model for multi-band photometry observations
	"""
	def __init__(self, arg):
		self.arg = arg

	def __call__(self, p, *args, **kwargs):
		return self.lnlike(p, *args, **kwargs) + self.lnprior(p)		


def do_sampling(args, phot)

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

   args = parser.parse_args()

   do_sampling(args, phot)

   return 0

if __name__ == "__main__":
    sys.exit(main())