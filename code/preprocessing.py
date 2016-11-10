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
    u = obs["MAG_U"] 
    g = obs["MAG_G"]
    r = obs["MAG_R"]
    i = obs["MAG_I"]
    z = obs["MAG_Z"]
    y = obs["MAG_Y"]
    Z = true['Z']    
    R = true['SIZE']    
    
    """ errors """

    ge = obs["MAGERR_G"]
    re = obs["MAGERR_R"]
    ie = obs["MAGERR_I"]
    ze = obs["MAGERR_Z"]
    ye = obs["MAGERR_Y"]
    
    CAT = np.vstack([u-g,g-r,r-i,i-z,z-y,r,Z,R]).T
    CATERR = np.vstack([ge,re,ie,ze,ye]).T
    CATERR = np.log(CATERR)
    
    """ filtering the catalogs """
  
    CATERR = CATERR[inds, :]
    zcut = np.where((CAT[inds,-2] < 1.8))[0]
    CATERR = CATERR[zcut]
    
    CAT = CAT[inds, :]
    CAT = CAT[CAT[:,-2] < 1.8 , :]
   
 
    """ some magnitudes are higher than 30, let's get rid of them """

    safe_mags = np.where((CAT[:,0] < 30)&(CAT[:,1] < 30)&(CAT[:,2] < 30)&(CAT[:,3] < 30)&(CAT[:,4] < 30)&(galprops["mass"]<10**13.3))[0] #(CAT[:,3] < 30)and(CAT[:,4] < 30)and(CAT[:,5] < 30))[0]
    CAT = CAT[safe_mags]
    galprops = galprops[safe_mags]
    CATERR = CATERR[safe_mags]

    """ putting together the final catalog before regression """
      
    ug, gr, ri, iz, zy, r, Z , R = CAT.T
    
    mass , sf = galprops["mass"], galprops["b300"]
    
    masterX = np.vstack([ug, gr, ri, iz, zy, r]).T
    masterY = np.vstack([Z, np.log10(mass), R]).T    

    length = masterX.shape[0]
    train_ind = np.random.choice(length , int(0.5 * length))
    test_ind = np.setdiff1d(np.arange(length) , train_ind)

    #np.savetxt(util.dat_dir()+"Xerr_train.dat", CATERR[train_ind]) 
    #np.savetxt(util.dat_dir()+"Xerr_test.dat", CATERR[test_ind]) 
    np.savetxt(util.dat_dir()+"X_train.dat", masterX[train_ind])
    np.savetxt(util.dat_dir()+"X_test.dat", masterX[test_ind])
    np.savetxt(util.dat_dir()+"Y_train.dat", masterY[train_ind])
    np.savetxt(util.dat_dir()+"Y_test.dat", masterY[test_ind])
 
    return None 


if __name__ == '__main__':

   data_vectors() 
