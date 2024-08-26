#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.fftpack import fft2,ifft2,fftshift

import bezier_optim as bo

def get_endpoints(num):
    """
    Link the imagen umber with the coordinates of endpoinds.
    Those were determined by hand.
    """
    if num==1:
        P_start=np.array([38,46])
        P_end = np.array([65,43])
    elif num==2:
        P_start=np.array([27,45])
        P_end = np.array([71,48])
    elif num==3:
        P_start=np.array([20,22])
        P_end = np.array([85,70])
    elif num==4:
        P_start=np.array([62,12])
        P_end = np.array([70,84])
    elif num==5:
        P_start=np.array([62,32])
        P_end = np.array([30,70])
    elif num==6:
        P_start=np.array([40,20])
        P_end = np.array([28,39])
    elif num==7:
        P_start=np.array([60,40])
        P_end = np.array([13,34])
    elif num==8:
        P_start=np.array([17,45])
        P_end = np.array([20,74])
    elif num==9:
        P_start=np.array([16,24])
        P_end = np.array([11,54])
    elif num==10:
        P_start=np.array([65,35])
        P_end = np.array([32,72])
    elif num==11:
        P_start=np.array([38,55])
        P_end = np.array([73,86])
    elif num==12:
        P_start=np.array([10,63])
        P_end = np.array([62,42])
    elif num==13:
        P_start=np.array([51,5])
        P_end = np.array([80,40])
    elif num==14:
        P_start=np.array([5,80])
        P_end = np.array([92,44])
    elif num==15:
        P_start=np.array([9,30])
        P_end = np.array([92,29])
    elif num==16:
        P_start=np.array([58,19])
        P_end = np.array([65,70])
    elif num==17:
        P_start=np.array([27,19])
        P_end = np.array([94,85])
    elif num==18:
        P_start=np.array([52,55])
        P_end = np.array([76,79])
    elif num==19:
        P_start=np.array([23,20])
        P_end = np.array([17,50])
    elif num==20:
        P_start=np.array([10,10])
        P_end = np.array([37,80])
    else:
        print("nonexisting number!")
    return P_start,P_end


def disp_results(Y_orig,Y,theta_est,X_est,binary,X_GT,P_start,P_end,curve_type='bezier'):
    """Display hte optim result"""
    P, Q = Y_orig.shape
    dy,dx = np.mgrid[0:P,0:Q]
    dx_flat = dx.flatten().reshape(-1,1)
    dy_flat = dy.flatten().reshape(-1,1)
    dt = np.linspace(0,1,100).reshape(-1,1)

    
    plt.figure(figsize=(17,3))

    plt.subplot(1,5,1)
    plt.imshow(Y_orig)
    plt.colorbar()
    plt.title('Observation')
    
    plt.subplot(1,5,2)
    plt.imshow(Y)
    plt.colorbar()
    plt.title('Gabor filt. amplitude')

    plt.subplot(1,5,3)
    P,Q = Y_orig.shape
    bo.disp_bezier(theta_est, 'w','k',dx_flat,dy_flat,dt,P,Q,curve_type,extent=None)
    #plt.ylim(P_start[0]-5,P_end[0]+5)
    #plt.xlim(P_start[1]-5,P_end[1]+5)
    plt.title('Estimated')
    plt.colorbar()

    plt.subplot(1,5,4)
    plt.imshow(Y)
    plt.colorbar()
    plt.title('Recap')

    if curve_type=='bezier':
        bezier = bo.get_bezier_curve(theta_est,dt)
    else:
        bezier = bo.get_segment_curve(theta_est,dt)
    #bezier = bo.get_bezier_curve(theta_est,dt)
    # shape = bo.get_bezier_image(theta_est, dx_flat,dy_flat,dt,P,Q,curve_type)

    n_pt = int((theta_est.size - 2)/2)

    #plt.imshow(shape)
    plt.plot(bezier[:,1],bezier[:,0],'--k')

    mask = binary*1.0
    mask[mask==1] = np.nan
    plt.imshow(mask,cmap=plt.cm.gray,alpha=0.75);

    points = np.reshape(theta_est[2:], (n_pt, 2))
    for i in range(1,n_pt-1):
        plt.plot(points[i,1],points[i,0],'xr')

    plt.plot(P_start[1],P_start[0],'*b',ms=10)
    plt.plot(P_end[1],P_end[0],'*b',ms=10)

    plt.subplot(1,5,5)
    plt.imshow(X_GT,cmap=plt.cm.gray,alpha=0.75)
    plt.title('Ground truth')
    plt.tight_layout()
    plt.show()

    P,Q = 100,100


def get_gabor_amplitude(Y):
    """ Gabor filtering, used for preprocessing."""
    # prepare filter bank kernels
    P,Q = Y.shape
    from skimage.filters import gabor_kernel
    kernels = []
    for theta in range(12):
        theta = theta / 12. * np.pi
        for sigma in (1,):
            for frequency in (0.2,):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
            #plt.imshow(kernel);plt.show()
    feat = np.zeros(shape=(len(kernels),P,Q))
    for k in range(len(kernels)):
        # scipy ndimage
        feat[k] = ndi.convolve(Y, kernels[k], mode='wrap')   
    return feat.mean(axis=0)

def get_image_endpoints(num,with_mask=True):

    binary = plt.imread(f"./first step/bin{num}.png")[::-1,:,0]==0

    dat=  np.load(f"./observations/y{num}.npz")
    Y_orig = dat.f.Y

    Y_orig_ft = fft2(Y_orig*1.0)
    Yo_fil = fftshift(Y_orig_ft)
    w = 5
    Yo_fil[50-w:50+w+1,50-w:50+w+1]=0
    Yfil = ifft2(fftshift(Yo_fil)).real

    Y =   get_gabor_amplitude(Yfil)

    Y = (Y - np.median(Y))/Y.std()

    P_start,P_end = get_endpoints(num)

    Y =   get_gabor_amplitude(Yfil)

    Y = (Y - np.median(Y))/Y.std()

    if with_mask:
        binblu = ndi.gaussian_filter((1-binary)*1.0,sigma=1)
        mask_binblu =binblu>binblu.max()/10

        Y[mask_binblu]=Y.min()
    
    
    return Y, Y_orig,P_start,P_end,binary
