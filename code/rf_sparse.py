import george
import data
import util 
import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
from matplotlib import gridspec
from sklearn.ensemble import RandomForestClassifier

RS = RobustScaler()

def data_vectors():

    X_test = np.loadtxt(util.dat_dir()+"X_test.dat")
    X_train = np.loadtxt(util.dat_dir()+"X_train.dat")
   
    Xerr_test = np.loadtxt(util.dat_dir()+"Xerr_test.dat")
    Xerr_train = np.loadtxt(util.dat_dir()+"Xerr_train.dat")
    
    Y_test = np.loadtxt(util.dat_dir()+"Y_test.dat")
    Y_train = np.loadtxt(util.dat_dir()+"Y_train.dat")
    
    return X_test, X_train, Y_test, Y_train


def scaler():

    X_test, X_train, Y_test, Y_train = data_vectors()
    train_X = RS.fit_transform(X_train)
    train_Y = Y_train
    test_X  = RS.transform(X_test)
    test_Y  = Y_test   
    
    return train_X, train_Y, test_X, test_Y    


def rf(num_bins, num_estimators):
    
    """
    random forrest classifier for determining p(z)'s.
    inputs:
    num_bins = number of redshift bins,
    num_estimators = number of estimators in scikit-learn random forrest classifier,
    outputs:
    label probabilities + best predictions + feature importance
    """
    """ Binning the redshifts"""
   
    X_train, Y_train, test_X, test_Y = scaler()
   
    nbins = num_bins
   
    #zmin, zmax = min(Y_train[:,0]), max(Y_train[:,0])
    #Mmin, Mmax = min(Y_train[:,1]), max(Y_train[:,1])
   
    zmin, zmax = 0., 2.
    Mmin, Mmax = 8.,13.5
    
    #print Mmin , Mmax, zmin, zmax
 
    zgrid = np.linspace(zmin, zmax, nbins)
    Mgrid = np.linspace(Mmin, Mmax, nbins)

    binsize1 = (zmax - zmin)/ nbins
    binsize2 = (Mmax - Mmin)/ nbins
    
    L = np.zeros((len(Y_train)))
    label_list = [] 
    """ Labeling the bins """
    for i in range(nbins):
        for j in range(nbins):
            ij = np.where((Y_train[:,0]>=zmin+i*binsize1)&(Y_train[:,0]<zmin+(i+1)*binsize1)&(Y_train[:,1]>=Mmin+j*binsize2)&(Y_train[:,1]<Mmin+(j+1)*binsize2))[0]
            if (len(ij) != 0 ):
               label_list.append([i,j])
            L[ij] = i + j * nbins
    
    """ Setting up the RF classifier"""

    clf = RandomForestClassifier(n_estimators=num_estimators, max_depth=None, min_samples_split=1, random_state=0)
    """ training """
    clf.fit(X_train, L)
    """feature importance """
    fi = clf.feature_importances_
    """best predictions"""
    Y_pred = clf.predict(test_X)
    """label probabilities"""
    prob = clf.predict_proba(test_X)
    """transformation quantities """
    trans = zmin , binsize1, Mmin, binsize2

    return prob , Y_pred, trans, label_list


def plot_distribution():

    magerrs , mags, masterX, masterY  = data_vectors()

    fig = plt.figure(1, figsize=(24,8))
    gs = gridspec.GridSpec(1,3)
    ax = plt.subplot(gs[0])
    sns.distplot(masterY[:,0], kde=True, hist=False, label='Redshift')
    #ax.set_xlim([0, 2.0])
    ax.set_title('Redshift distrubtion', fontsize=18)
    ax.legend(fontsize=13)
    ax.set_xlabel('Redshift', fontsize=18)    
    ax = plt.subplot(gs[1])
    sns.distplot(masterY[:,1], kde=True, hist=False, label=r'$M_{\star}$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title('stellar mass distrubtion', fontsize=18)
    ax.legend(fontsize=13)
    ax.set_xlabel('stellar mass', fontsize=18)    
    ax = plt.subplot(gs[2])
    sns.distplot(masterY[:,2], kde=True, hist=False, label=r'$SFR$')
    ax.set_xscale("log")
    ax.set_title('size', fontsize=18)
    ax.legend(fontsize=13)
    ax.set_xlabel('R', fontsize=18)    
    fig.savefig("z.pdf" , box_inches = "tight")
    
    fig = plt.figure(1, figsize=(8,8))
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])
    sns.distplot(mags[:,0], kde=True, hist=False, label = '')
    sns.distplot(mags[:,1], kde=True, hist=False, label = '')
    sns.distplot(mags[:,2], kde=True, hist=False, label = '')
    sns.distplot(mags[:,3], kde=True, hist=False, label = '')
    sns.distplot(mags[:,4], kde=True, hist=False, label = '')
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
    ax.set_title('magerrs', fontsize=18)
    ax.legend(fontsize=13)
    ax.set_xlabel('magerrs', fontsize=18)    
    fig.savefig("magerrs.pdf" , box_inches = "tight")

    return None


if __name__ == '__main__':

    from matplotlib.colors import LogNorm
    
    X_train, Y_train, test_X, test_Y = scaler()
    
    prob , bestfit , trans, label_list = rf(num_bins =10, num_estimators=300) 
    zmin , binsize1, Mmin, binsize2 = trans


    for i in range(50):
       
       prob2d = np.zeros((10,10))
       

       for k in xrange(len(prob[i])):
           
           prob2d[label_list[k][1] , label_list[k][0]] = prob[i][k]

       plt.figure(figsize=(10,10)) 
       image = prob2d 
       image = image / np.sum(image)    
       
       im = plt.imshow(prob2d , interpolation = "none", origin = "lower", cmap = plt.cm.viridis, extent = [0,2,8,13.5], aspect = (2-0)/(13.5-8))
       plt.plot(test_Y[i,0], test_Y[i,1] , "k*")
       plt.xlabel("$z$")
       plt.ylabel("$M_{\star}$")
       plt.xlim([0,2])
       plt.ylim([8,13.5])
       plt.title(r"$P(z,M_{\star})$" , fontsize = 20)
       plt.colorbar()
       plt.savefig(util.fig_dir()+str(i)+".png")
       plt.close()
