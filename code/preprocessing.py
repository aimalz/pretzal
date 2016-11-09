import george
from george.kernels import ExpKernel , WhiteKernel
import data
import util 
import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
from matplotlib import gridspec

RS = RobustScaler()

def data_vectors():

    """load the additional galaxy properties"""
 
    galprops = data.generate_alpha()

    """load the data vectors"""
    
    inds = data.filter()
    true = data.load_truth()
    obs = data.load_observations()
    true = data.load_truth()
   
    """ obsevations """
    
    g = obs["MAG_G"]
    r = obs["MAG_R"]
    i = obs["MAG_I"]
    z = obs["MAG_Z"]
    y = obs["MAG_Y"]
    Z = true['Z']    
    
    """ errors """

    ge = obs["MAGERR_G"]
    re = obs["MAGERR_R"]
    ie = obs["MAGERR_I"]
    ze = obs["MAGERR_Z"]
    ye = obs["MAGERR_Y"]
    
    CAT = np.vstack([g,r,i,z,y,Z]).T
    CATERR = np.vstack([ge,re,ie,ze,ye]).T
    CATERR = np.log(CATERR)
    
    """ filtering the catalogs """
  
    CATERR = CATERR[inds, :]
    zcut = np.where((CAT[inds,-1] < 1.8))[0]
    CATERR = CATERR[zcut]
    
    CAT = CAT[inds, :]
    CAT = CAT[CAT[:,-1] < 1.8 , :]
   
 
    """ some magnitudes are higher than 30, let's get rid of them """

    safe_mags = np.where((CAT[:,0] < 30)&(CAT[:,1] < 30)&(CAT[:,2] < 30)&(CAT[:,3] < 30)&(CAT[:,4] < 30)&(galprops["mass"]<10**13.3))[0] #(CAT[:,3] < 30)and(CAT[:,4] < 30)and(CAT[:,5] < 30))[0]
    CAT = CAT[safe_mags]
    galprops = galprops[safe_mags]
    CATERR = CATERR[safe_mags]

    """ putting together the final catalog before regression """
      
    g, r, i, z, y, Z = CAT.T
    gr, ri, iz, zy = g-r, r-i, i-z, z-y
    
    mass , sf = galprops["mass"], galprops["b300"]
    
    mags = np.vstack([g,r,i,z,y]).T
    masterX = np.vstack([gr, ri, iz, zy, y]).T
    masterY = np.vstack([Z, mass, sf]).T    

    length = mags.shape[0]
    train_ind = np.random.choice(length , int(0.5 * length))
    test_ind = np.setdiff1d(np.arange(length) , train_ind)

    np.savetxt(util.dat_dir()+"Xerr_train.dat", CATERR[train_ind]) 
    np.savetxt(util.dat_dir()+"Xerr_test.dat", CATERR[test_ind]) 
    np.savetxt(util.dat_dir()+"X_train.dat", mags[train_ind])
    np.savetxt(util.dat_dir()+"X_test.dat", mags[test_ind])
    np.savetxt(util.dat_dir()+"Y_train.dat", masterY[train_ind])
    np.savetxt(util.dat_dir()+"Y_test.dat", masterY[test_ind])
 
    return CATERR , mags, masterX, masterY


if __name__ == '__main__':

   data_vectors() 
