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
    g = obs["OMAG"][:,0]
    r = obs["OMAG"][:,1]
    i = obs["OMAG"][:,2]
    z = obs["OMAG"][:,3]
    y = obs["OMAG"][:,4]

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

    CAT = np.vstack([g-r,r-i,i-z,z-y,r+dm,Z,R,COEFFS]).T
    CATERR = np.vstack([ge,re,ie,ze,ye]).T
    CATERR = np.log(CATERR)

    """ filtering the catalogs """

    CATERR = CATERR[inds, :]
    zcut = np.where((CAT[inds,-3] < 2.))[0]
    CATERR = CATERR[zcut]

    CAT = CAT[inds, :]
    CAT = CAT[CAT[:,-3] < 2. , :]


    """ some magnitudes are higher than 30, let's get rid of them """

    safe_mags = np.where((CAT[:,0] < 30)&(CAT[:,1] < 30)&(CAT[:,2] < 30)&(CAT[:,3] < 30)&(CAT[:,4] < 30)&(galprops["mass"]<10**13.3))[0] #(CAT[:,3] < 30)and(CAT[:,4] < 30)and(CAT[:,5] < 30))[0]
    CAT = CAT[safe_mags]
    galprops = galprops[safe_mags]
    CATERR = CATERR[safe_mags]

    """ putting together the final catalog before regression """

    gr, ri, iz, zy, r, Z , R = CAT.T

    mass , sf = galprops["mass"], galprops["b300"]

    masterX = np.vstack([gr, ri, iz, zy, r]).T
    masterY = np.vstack([Z, np.log10(mass), R]).T

    length = masterX.shape[0]
    train_ind = np.random.choice(length , int(0.5 * length))
    test_ind = np.setdiff1d(np.arange(length) , train_ind)

    #np.savetxt(util.dat_dir()+"Xerr_train.dat", CATERR[train_ind])
    #np.savetxt(util.dat_dir()+"Xerr_test.dat", CATERR[test_ind])
    np.savetxt(util.dat_dir()+"X_train_A.dat", masterX[train_ind])
    np.savetxt(util.dat_dir()+"X_test_A.dat", masterX[test_ind])
    np.savetxt(util.dat_dir()+"Y_train_A.dat", masterY[train_ind])
    np.savetxt(util.dat_dir()+"Y_test_A.dat", masterY[test_ind])

    return None


if __name__ == '__main__':

   data_vectors()
