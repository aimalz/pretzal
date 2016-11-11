import data
import util 
import numpy as np

def data_vectors():

    """load the data vectors"""
    
    inds = data.filter()
    true = data.load_truth()
    obs = data.load_observations()
    true = data.load_truth()
   
    """ multiband photometry """

    u = obs["MAG_U"] 
    g = obs["MAG_G"]
    r = obs["MAG_R"]
    i = obs["MAG_I"]
    z = obs["MAG_Z"]
    y = obs["MAG_Y"]
    Z = true['Z']    
    
    """ multiband photometry error"""

    
    ge = obs["MAGERR_G"]
    ue = ge               #hardcoded: the catalog doesn't have u_err
    re = obs["MAGERR_R"]
    ie = obs["MAGERR_I"]
    ze = obs["MAGERR_Z"]
    ye = obs["MAGERR_Y"]
    Zerr = obs['PHOTOZ_GAUSSIAN']
   
    CAT = np.vstack([u, g, r, i, z, y, Z]).T 
    CATERR = np.vstack([ue, ge, re, ie, ze, ye, Zerr]).T
    
    """ filtering the catalogs """
  
    CATERR = CATERR[inds, :]
    zcut = np.where((CAT[inds,-2] < 1.8))[0]
    CATERR = CATERR[zcut]
    
    CAT = CAT[inds, :]
    CAT = CAT[CAT[:,-2] < 1.8 , :]
   
 
    """ some magnitudes are higher than 30, let's get rid of them """

    safe_mags = np.where((CAT[:,0] < 30)&(CAT[:,1] < 30)&(CAT[:,2] < 30)&(CAT[:,3] < 30)&(CAT[:,4] < 30)&(CAT[:,5] < 30))[0]
    CAT = CAT[safe_mags]
    CATERR = CATERR[safe_mags]

    """ putting together the final catalog before regression """
     
    np.savetxt(util.dat_dir()+"magZ.dat", CAT)
    np.savetxt(util.dat_dir()+"magZerr.dat", CATERR)
 
    return None 


if __name__ == '__main__':

   data_vectors() 
