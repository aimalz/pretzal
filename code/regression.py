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
import GPy

RS = RobustScaler()

def data_vectors():

    X_test = np.loadtxt(util.dat_dir()+"X_test.dat")
    X_train = np.loadtxt(util.dat_dir()+"X_train.dat")
   
    X_test = np.loadtxt(util.dat_dir()+"Xerr_test.dat")
    X_train = np.loadtxt(util.dat_dir()+"Xerr_train.dat")
    
    Y_test = np.loadtxt(util.dat_dir()+"X_test.dat")
    Y_train = np.loadtxt(util.dat_dir()+"X_train.dat")
    
    return X_test, X_train, Y_test, Y_train


def scaler():

    X_test, X_train, Y_test, Y_train = data_vectors()

    train_X = RS.fit_transform(X_train)
    train_Y = Y_train
    test_X  = RS.transform(X_test)
    test_Y  = Y_test   
    
    return train_X, train_Y, test_X, test_Y    


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

    
   #fraction = 0.6
   #random_spliter(fraction)
   #plot_distribution()
   GPregression()
