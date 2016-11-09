#from george import kernel
import data
import util 
import numpy as np
from sklearn.preprocessing import RobustScaler
#%matplotlib inline
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
    
    #u = obs["MAG_U"]
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
    return CATERR , mags, masterX, masterY

def random_spliter(fractions):

    CATERR , mags , masterX, masterY  = data_vectors()
    half = int(fraction * len(masterY))
    train_X = RS.fit_transform(masterX[:half,:])
    train_Y = masterY[:half,:]
    test_X  = RS.transform(masterX[half:,:])
    test_Y  = masterY[half:,:]    

    return train_X, train_Y, test_X, test_Y    

def plot_distribution():

    magerrs , mags, masterX, masterY  = data_vectors()

    fig = plt.figure(1, figsize=(24,8))
    gs = gridspec.GridSpec(1,3)
    ax = plt.subplot(gs[0])
    sns.distplot(masterY[:,0], kde=True, hist=False, label='Redshift')
    ax.set_xlim([0, 2.0])
    ax.set_title('Redshift distrubtion', fontsize=18)
    ax.legend(fontsize=13)
    ax.set_xlabel('Redshift', fontsize=18)    
    ax = plt.subplot(gs[1])
    sns.distplot(masterY[:,1], kde=True, hist=False, label=r'$M_{\star}$')
    #ax.set_xlim([0, 2.0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title('stellar mass distrubtion', fontsize=18)
    ax.legend(fontsize=13)
    ax.set_xlabel('stellar mass', fontsize=18)    
    ax = plt.subplot(gs[2])
    sns.distplot(masterY[:,2], kde=True, hist=False, label=r'$SFR$')
    #ax.set_xlim([0, 2.0])
    ax.set_xscale("log")
    ax.set_title('stellar formation rate', fontsize=18)
    ax.legend(fontsize=13)
    ax.set_xlabel('SFR', fontsize=18)    
    fig.savefig("z.pdf" , box_inches = "tight")
    
    fig = plt.figure(1, figsize=(8,8))
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])
    sns.distplot(mags[:,0], kde=True, hist=False, label = '')
    sns.distplot(mags[:,1], kde=True, hist=False, label = '')
    sns.distplot(mags[:,2], kde=True, hist=False, label = '')
    sns.distplot(mags[:,3], kde=True, hist=False, label = '')
    sns.distplot(mags[:,4], kde=True, hist=False, label = '')
    #ax.set_xlim([0, 2.0])
    ax.set_title('magnitude distrubtion', fontsize=18)
    ax.legend(fontsize=13)
    ax.set_xlabel('magnitudes', fontsize=18)    
    fig.savefig("mags.pdf" , box_inches = "tight")


    fig = plt.figure(1, figsize=(8,8))
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])
    sns.distplot(magerrs[:,0], kde=True, hist=False, label = '')
    sns.distplot(magerrs[:,1], kde=True, hist=False, label = '')
    sns.distplot(magerrs[:,2], kde=True, hist=False, label = '')
    sns.distplot(magerrs[:,3], kde=True, hist=False, label = '')
    sns.distplot(magerrs[:,4], kde=True, hist=False, label = '')
    #ax.set_xlim([0, 2.0])
    ax.set_title('magerrs', fontsize=18)
    ax.legend(fontsize=13)
    ax.set_xlabel('magerrs', fontsize=18)    
    fig.savefig("magerrs.pdf" , box_inches = "tight")

    return None

if __name__ == '__main__':
 
   fraction = 0.6
   random_spliter(fraction)
   plot_distribution()
