#from george import kernel
import data
import util 
import numpy as np
from sklearn import preprocessing

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
    
    CAT = np.vstack([u,g,r,i,z,y,Z]).T
       
    """ filtering the catalogs """
    CAT = CAT[inds, :]
    CAT = CAT[CAT[:,-1] < 1.8 , :]
     
    u, g, r, i, z, y, Z = CAT.T
    ug, gr, ri, iz, zy = u-g, g-r, r-i, i-z, z-y
    mass , sf = galprops["mass"], galprops["b300"]
    
    masterX = np.rec.fromarrays(np.vstack([ug, gr, ri, iz, zy, y]), 
                               names = ["ug", "gr", "ri", "iz", "zy", "y"])
    masterY = np.rec.fromarrays(np.vstack([Z, mass, sf]), 
                               names = ["Z", "mass", "sf"])
    
    return masterX, masterY

def random_spliter(fractions):

    masterX, masterY  = data_vectors()

    half = int(fraction * len(masterY))
    X_scaled = preprocessing.scale(masterX)
    train_X = X_scaled[:half,:]
    train_Y = masterY[:half]
    test_X = X_scaled[half:,:]
    test_Y = masterY[half:]    

    return train_X, train_Y, test_X, test_Y    


if __name__ == '__main__':
 
   fraction = 0.6
   random_spliter(fraction)
