import george
from george.kernels import ExpKernel , WhiteKernel
import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
from matplotlib import gridspec
import scipy.interpolate as spi

import AardvarkData as data
import util

RS = RobustScaler()

def data_vectors():

    """load the additional galaxy properties"""

    galprops = data.generate_alpha()

    """load the data vectors"""

    inds = data.filter()
    true = data.load_truth()
    obs = data.load_observations()

    """ obsevations """
    g = obs["TMAG"][:,0]
    r = obs["TMAG"][:,1]
    i = obs["TMAG"][:,2]
    z = obs["TMAG"][:,3]
    y = obs["TMAG"][:,4]

    """ errors """

    dm = obs["DELTAM"]

    ge = obs["OMAGERR"][:,0]
    re = obs["OMAGERR"][:,1]
    ie = obs["OMAGERR"][:,2]
    ze = obs["OMAGERR"][:,3]
    ye = obs["OMAGERR"][:,4]

    """ inferrables """

    Z = true['Z']
    R = true['SIZE']
    COEFFS = true['COEFFS']
        
    CAT = np.vstack([g-r,r-i,i-z,z-y,r,dm,Z,R]).T
    CATERR = np.vstack([ge,re,ie,ze,ye]).T
    CATERR = np.log(CATERR)

    """ filtering the catalogs """

    CATERR = CATERR[inds, :]
    zcut = np.where((CAT[inds,-2] < 2.))[0]
    CATERR = CATERR[zcut]

    CAT = CAT[inds, :]
    CAT = CAT[CAT[:,-2] < 2. , :]

    COEFFS = COEFFS[inds, :]
    COEFFS = COEFFS[zcut]

    """ need to correct the r-band magnitudes so they match the coefficients """

    (r_lambda, r_filter) = data.load_filter()
    templates = data.load_templates()
    wavelengths = data.load_wavelengths()

    difs = wavelengths[1:]-wavelengths[:-1]

    ind_lo = np.where(wavelengths < max(r_lambda))
    ind_hi = np.where(wavelengths > min(r_lambda))
    ind = np.intersect1d(ind_lo, ind_hi)

    interpolator = spi.interp1d(r_lambda, r_filter)
    r_filter_matched = interpolator(wavelengths[ind])

    t_flux = np.array([3.34e4*wavelengths[ind]**2*templates[b][ind] for b in range(5)])
    t_r_flux = np.array([np.dot(r_filter_matched*t_flux[b], difs[ind]) for b in range(5)])
    r_flux = np.array([np.dot(COEFFS[i], t_r_flux) for i in range(len(zcut))])
    r_mags = -2.5*np.log10(r_flux)+8.9

    #""" some magnitudes are higher than 30, let's get rid of them """

    #safe_mags = np.where((CAT[:,0] < 30)&(CAT[:,1] < 30)&(CAT[:,2] < 30)&(CAT[:,3] < 30)&(galprops["mass"]<10**13.3))[0] #(CAT[:,3] < 30)and(CAT[:,4] < 30)and(CAT[:,5] < 30))[0]
    #CAT = CAT[safe_mags]
    #galprops = galprops[safe_mags]
    #CATERR = CATERR[safe_mags]

    """ putting together the final catalog before regression """

    gr, ri, iz, zy, r, dm, Z , R = CAT.T

    mass , sf = galprops["mass"], galprops["b300"]

    masterX = np.vstack([gr, ri, iz, zy, r_mags+dm]).T
    masterY = np.vstack([Z, np.log10(mass), R]).T

    length = masterX.shape[0]
    train_ind = np.random.choice(length , int(0.5 * length))
    test_ind = np.setdiff1d(np.arange(length) , train_ind)

    #np.savetxt(util.dat_dir()+"Xerr_train.dat", CATERR[train_ind])
    #np.savetxt(util.dat_dir()+"Xerr_test.dat", CATERR[test_ind])
    np.savetxt(util.dat_dir()+"X_train_A.dat", masterX[train_ind])
    np.savetxt(util.dat_dir()+"X_test_A.dat", masterX[test_ind])
    np.savetxt(util.dat_dir()+"COEFFS_train_A.dat", COEFFS[train_ind])
    np.savetxt(util.dat_dir()+"COEFFS_test_A.dat", COEFFS[test_ind])
    np.savetxt(util.dat_dir()+"Y_train_A.dat", masterY[train_ind])
    np.savetxt(util.dat_dir()+"Y_test_A.dat", masterY[test_ind])

    return None


if __name__ == '__main__':

   data_vectors()
