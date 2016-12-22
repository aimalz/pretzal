"""
Module for handling the data:
"""
import pyfits as pf
import numpy as np
import util
from astropy.cosmology import FlatLambdaCDM

def filter():
    '''
    generate indices for downsampling of the data
    '''
    np.random.seed(12345)
    N = 100000 # hard-coded number for downsampling the galaxies
    indices = np.random.choice(5000000, N)

    return indices

def load_truth():
    ''' loads the Aardvark catalog containing the true galaxy properties
    '''
    data_file = ''.join([util.dat_dir(),
            'Aardvark-0.0.fits'])

    return pf.open(data_file)[1].data

def load_observations():

    ''' loads the Aardvark catalog containing the observed galaxy properties
    '''
    data_file = ''.join([util.dat_dir(),
            'Aardvark-0_DECAM.0.fits'])

    return pf.open(data_file)[1].data

def load_filter():

    ''' loads the SDSS R-band filter curve
    '''
    data_file = ''.join([util.dat_dir(),
                         'sdss_r0.par'])

    with open(data_file) as csvfile:
        tuples = (line.split(None) for line in csvfile)
        rfilter = [[pair[k] for k in range(1,len(pair))] for pair in tuples]
        rfilter = np.array([[np.float(r_f) for r_f in line] for line in rfilter[74:]])
    r_lambda = rfilter.T[0].T
    r_filter = rfilter.T[3].T

    return (r_lambda, r_filter)

def load_templates():

    ''' loads the kcorrect templates
    '''
    data_file = ''.join([util.dat_dir(),
                         'vmatrix.default.dat'])

    with open(data_file) as csvfile:
        tuples = (line.split(None) for line in csvfile)
        templates = [np.float(pair[0]) for pair in tuples]
        templates = np.array(templates[1:])
        templates = np.array(np.split(templates, 5))

    return templates

def load_wavelengths():

    ''' loads the wavelengths for the kcorrect templates
    '''
    data_file = ''.join([util.dat_dir(),
                         'lambda.default.dat'])

    with open(data_file) as csvfile:
        tuples = (line.split(None) for line in csvfile)
        wavelengths = [np.float(pair[0]) for pair in tuples]
        wavelengths = np.array(wavelengths[1:])

    return wavelengths 

def generate_alpha():
    '''generates additional galaxy properties : Mstar, SSFR, ...
       and write them into a file
    '''
    data_file = ''.join([util.dat_dir(),
            'Aardvark-0.0.fits'])
    indices = filter()

    redshifts = pf.open(data_file)[1].data['Z'][indices]
    coeffs = pf.open(data_file)[1].data['COEFFS'][indices]
    coeffs = coeffs[redshifts < 2.]
    coeffs = coeffs.T
    redshifts = redshifts[redshifts < 2.]
    nredshifts = len(redshifts)

    specfile =  ''.join([util.dat_dir(),
            'k_nmf_derived.default.fits'])
    t = pf.open(specfile)
    tbdata=t[18].data
    tmass=t[16].data
    tmremain=t[17].data
    tmetallicity=t[18].data
    tmass300=t[19].data
    tmass1000=t[20].data

    #initialize arrays
    b300=np.zeros(nredshifts)
    b1000=np.zeros(nredshifts)
    tmp_mass=np.zeros(nredshifts)
    mets=np.zeros(nredshifts)
    mass=np.zeros(nredshifts)
    intsfh=np.zeros(nredshifts)

    cosmology=FlatLambdaCDM(H0=70, Om0=0.286)
    dmod=cosmology.distmod(redshifts)
    # do all the calculations, looping over objects
    for i in np.arange(nredshifts):
        b300[i]=np.sum(tmass300*coeffs[:,i])/np.sum(tmass*coeffs[:,i])
        b1000[i]=np.sum(tmass1000*coeffs[:,i])/np.sum(tmass*coeffs[:,i])
        tmp_mass[i]=np.sum(tmremain*coeffs[:,i])
        mets[i]=np.sum(tmremain*tmetallicity*coeffs[:,i])/tmp_mass[i]
        mass[i]=tmp_mass[i]*10.**(0.4*dmod.value[i])
        intsfh[i]=np.sum(coeffs[:,i])*10.**(0.4*dmod.value[i])
    galprop =  np.rec.fromarrays(np.vstack([b300, b1000, tmp_mass, mets, mass, intsfh]), names = ["b300", "b1000", "tmp_mass", "mets", "mass", "intsfh"])

    return galprop
