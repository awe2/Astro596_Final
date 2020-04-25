import numpy as np
import matplotlib.pyplot as plt
import random
import os
import seaborn as sns
from scipy.stats import gaussian_kde



def show_galaxy(alias):
    if alias[0:2] == 'HP':
        a = np.load('..//DATA//HP_inputs//' + alias + '.npy')
    if alias[0:2] == 'MP':
        a = np.load('..//DATA//MP_inputs//' + alias + '.npy')
    if alias[0:2] == 'LP':
        a = np.load('..//DATA//LP_inputs//' + alias + '.npy')
    img = np.ones((120,120,3),dtype=float)
    img[:,:,0] = a[:,:,3]
    img[:,:,1] = a[:,:,2]
    img[:,:,2] = a[:,:,1]
    plt.imshow(img/255.0)
    plt.show()




n_classes=180
redshift_bin_middles = np.array(np.array(range(n_classes)) * 0.4/n_classes + 0.4/(n_classes*2)) 
def expected_values(Out,bins=redshift_bin_middles):
    """
    (Probabilities,bins)
    Provides the prediction from the softmax activation provided the middle value of each bin
    """
    Y = np.empty((np.shape(Out)[0]))
    for i in range(np.shape(Out)[0]):
        Y[i] = np.sum(bins * Out[i,:])
    return(Y)


def maximum_values(Out,high=0.4,n_classes=180):
    Y = high*((np.argmax(Out,1))/n_classes)+(high/n_classes*2)
    return(Y)



def visualize_performance(X,Y,title='title',string='plot.png',save=False):
    """
    (X,Y,title,string,save)
    X = array of true redshifts
    Y = array of photoZ predictions
    title = string title of graph
    string = what to title the file
    """
    residuals = ((X - Y)/ (1 + X))
    prediction_bias = np.mean(residuals)
    RMS = np.sqrt(np.mean(residuals**2))
    MAD = 1.4826*np.median(abs(residuals - np.median(residuals))) #median absolute deviation
    outlier_fraction_eta = len(residuals[residuals > 3*MAD*(1+X)])/len(residuals)
    outlier_fraction_eta_RMS = len(residuals[residuals > 3*RMS*(1+X)])/len(residuals)
    
    print('prediction bias (<z>) =', prediction_bias)
    print('Median Absolute Deviation (MAD) = ', MAD)
    print('Root Mean Square (RMS) = ', RMS)
    print('outlier fraction (eta) =', outlier_fraction_eta)
    print('outlier fraction (eta) RMS =', outlier_fraction_eta_RMS)

    plt.plot(X[residuals > 3*MAD],Y[residuals > 3*MAD],'g.')
    plt.plot([0.0,0.7],[0.0,0.7],'k--')

    plt.title(title)
    plt.xlabel('Spectro-Z')
    plt.ylabel('Photo-Z')
    plt.ylim(-0.01,0.4)
    plt.xlim(-0.01,0.4)

    plt.text(0.00,0.35,'(<{}z>) ='.format('\u0394') + str(round(prediction_bias,4)))
    plt.text(0.00,0.30,'(RMS) = ' + str(round(RMS,3)))
    plt.text(0.00,0.25,'(eta) = ' + str(round(outlier_fraction_eta_RMS,4)))

    ax = sns.kdeplot(X, Y, cmap="plasma", shade=True, shade_lowest=False, cbar=True)
    if save==True:
        plt.savefig(string)
    plt.show()


def Bias_of_residuals(X,Y,title='title',strng='plot.png',save=False):
    """
    X = spectro Z
    Y = photo Z
    """
    residuals = ((X - Y)/ (1 + X))
    prediction_bias = np.mean(residuals)
    MAD = 1.4826*np.median(abs(residuals - np.median(residuals))) #median absolute deviation
    outlier_fraction_eta_MAD = len(residuals[residuals > 3*MAD])/len(residuals)
    

    bins=np.linspace(-0.1,0.1,40)
    weights = np.ones_like(residuals)/float(len(residuals))
    n_out = plt.hist(residuals,bins,weights=weights)

    plt.title(title)
    plt.ylabel('Counts')
    plt.xlabel('Z error')
    plt.text(-0.1,0.12,'<z> = {}'.format(round(prediction_bias,5)))

    plt.vlines(0,0,1,color='k',linestyle='dashed',label='zero bias')
    plt.vlines(np.median(residuals),0,3000,color='r',linestyle='dashed',label='median')
    plt.ylim(0,0.20)

    kernel = gaussian_kde(residuals)
    bins=np.linspace(-0.1,0.1,200)
    gauss_out = kernel.evaluate(bins)/len(bins)
    plt.plot(bins,gauss_out)
    plt.legend()
    if save==True:
        plt.savefig(string)
    plt.show()



def Sample_PDF(X,Y,Out,index=0,title='title',string='plot.png',save=False,bins=redshift_bin_middles):
    """
    X = spectro Z
    Y = photoZ
    out=probabilities array
    bins=ins evaluated upon
    """
    #one PDF would be:
    residuals = ((X - Y)/ (1 + X))
    prediction_bias = np.mean(residuals)
    MAD = 1.4826*np.median(abs(residuals - np.median(residuals))) #median absolute deviation
    outlier_fraction_eta = len(residuals[residuals > 5*MAD])/len(residuals)

    PDF1 = Out[index,:]
    PDF2 = Out[index+1,:]
    PDF3 = Out[index+2,:]
    PDF4 = Out[index+3,:]
    PDF5 = Out[index+4,:]
    expected_value1 = np.sum(bins*PDF1)
    expected_value2 = np.sum(bins*PDF2)
    expected_value3 = np.sum(bins*PDF3)
    expected_value4 = np.sum(bins*PDF4)
    expected_value5 = np.sum(bins*PDF5)
    #lets plot it:
    plt.plot(bins,PDF1,label='PDF',color='r')
    plt.plot(bins,PDF2,label='PDF',color='b')
    plt.plot(bins,PDF3,label='PDF',color='k')
    plt.plot(bins,PDF4,label='PDF',color='g')
    #plt.plot(bins,PDF5,label='PDF',color='c')
    
    
    #plt.vlines([expected_value],color='k',linestyle='dashed',ymin=0,ymax=max(PDF)+0.02,label='Expected Value')
    plt.vlines([X[index]],color='r',linestyle='dashed',ymin=0,ymax=max(PDF1)+0.02,label='Spectro Z')
    plt.vlines([X[index+1]],color='b',linestyle='dashed',ymin=0,ymax=max(PDF2)+0.02,label='Spectro Z')
    plt.vlines([X[index+2]],color='k',linestyle='dashed',ymin=0,ymax=max(PDF3)+0.02,label='Spectro Z')
    plt.vlines([X[index+3]],color='g',linestyle='dashed',ymin=0,ymax=max(PDF4)+0.02,label='Spectro Z')
    #plt.vlines([X[index+4]],color='c',linestyle='dashed',ymin=0,ymax=max(PDF5)+0.02,label='Spectro Z')
    
    plt.ylim(0,max(PDF1)+0.02)
    plt.xlim(0,0.4)
    #plt.legend()
    plt.xlabel('Spectroscopic Redshift')
    plt.ylabel('Probability')
    plt.title(title)
    #plt.text(0.26,0.02,'Residual Error: {}'.format(abs(round(expected_value-X[index],4))))

    one_bin = 0.4/180
    gaussian1 = np.random.choice(a=bins,size=1000,p=PDF1)
    expected_value_new_maybe1 = np.mean(gaussian1)
    STD1 = np.std(gaussian1)

    gaussian2 = np.random.choice(a=bins,size=1000,p=PDF2)
    expected_value_new_maybe2 = np.mean(gaussian2)
    STD2 = np.std(gaussian2)    
    
    gaussian3 = np.random.choice(a=bins,size=1000,p=PDF3)
    expected_value_new_maybe3 = np.mean(gaussian3)
    STD = np.std(gaussian3)
    
    gaussian4 = np.random.choice(a=bins,size=1000,p=PDF4)
    expected_value_new_maybe4 = np.mean(gaussian4)
    STD = np.std(gaussian4)
    
    gaussian5 = np.random.choice(a=bins,size=1000,p=PDF5)
    expected_value_new_maybe5 = np.mean(gaussian5)
    STD = np.std(gaussian5)
    
    #plt.text(0.26,0.015,'Standard Error: {}'.format(round(STD,4)))
    #print('gaussian mean value: ', expected_value_new_maybe)
    #print('gaussian standard deviation: ', STD)
    #print('gaussian five sigma: ', 5*STD)
    
    if save==True:
        plt.savefig(string)

    plt.show()


def PIT(X,Out,title='title',string='plot.png',save=False,n_classes=180):
    """
    X = spectro Z array
    Out = probabilities array
    """
    print(np.shape(X))
    X_true_bin = np.round((X/0.4)*(n_classes-1),0).astype(int)
    X_true_bin[X_true_bin>=n_classes] = n_classes-1 
    PIT = []

    print(np.shape(X_true_bin))
    print(np.shape(Out))
    for i in range(len(X_true_bin)):
        PIT.append(np.sum((Out[i,:])[0:X_true_bin[i]]))

    PIT=np.asarray(PIT)

    n_bins=n_classes
    bins=np.linspace(0,1,n_bins)
    #draw the line that is if it was a perfect distribution. it would have..
    #len(PIT)/180 #number in each bin
    plt.hlines((len(PIT)/n_bins),0,1,colors='k',linestyles='solid')
    plt.xlim(0,1)
    plt.hist(PIT,bins)
    plt.title(title)
    if save==True:
        plt.savefig(string)
    plt.show()

    #catastropic outliers are thos with PIT values <0.0001 or >0.9999; a normal distribution would have 0.0002 fraction
    catastropic_outlier_fraction = (len(PIT[PIT<0.0001]) + len(PIT[PIT>0.9999])) / len(PIT)
    print(catastropic_outlier_fraction)
    print(' ')
    print("Normal distribution's fraction is 0.0002") 


