import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
import time
import math
import matplotlib.pyplot as plt
import os
import scipy.io as sio

Lx = 10*np.pi
Ly = 10*np.pi
Nx = 128
Ny = 128
xx = (Lx / Nx) * np.linspace(0, Nx, Nx)
yy = (Ly / Ny) * np.linspace(-Ny / 2 + 1, Ny / 2, Ny)
X, Y = np.meshgrid(xx, yy)


mu_tst = np.linspace(.001,.999,10)
for i in range(1,10):
    print("Generating Sample Plots ",i)
    theta = np.sqrt(1-mu_tst[i]**2)*X + np.log(2*np.cosh(mu_tst[i]*Y))
    w = np.cos(theta)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(w)
    ax[1].imshow(theta)
    plt.suptitle("RCN PGB Phase {}".format(i))
    plt.tight_layout()
    plt.show()


num_samples = 1000
mus = np.linspace(.01,.99,num_samples)
ws = np.zeros(shape=(num_samples,Ny,Nx))
thetas = np.zeros(shape=(num_samples,Ny,Nx))
wxs = np.zeros(shape=(num_samples,Ny,Nx))
wxxs = np.zeros(shape=(num_samples,Ny,Nx))
wys = np.zeros(shape=(num_samples,Ny,Nx))
wyys = np.zeros(shape=(num_samples,Ny,Nx))
for i in range(num_samples):
    print("Generating Data ", i)
    theta = np.sqrt(1-mus[i]**2)*X + np.log(2*np.cosh(mus[i]*Y))
    thetas[i, :, :] = theta
    w = np.cos(theta)
    ws[i, :, :] = w
    wx = -np.sqrt(1-mus[i]**2)*np.sin(theta)
    wxs[i,:,:] = wx
    wxx = (mus[i]**2-1)*np.cos(theta)
    wxxs[i,:,:] = wxx
    wy = -mus[i]*np.tanh(mus[i]*Y)*np.sin(theta)
    wys[i,:,:] = wy
    wyy = -mus[i]**2*np.tanh(mus[i]*Y)**2*np.cos(theta) - (mus[i]**2/(np.cosh(mus[i]*Y)**2))*np.sin(theta)
    wyys[i,:,:] = wyy

mdict = {"ws": ws,"thetas":thetas,"wxs":wxs,"wxxs":wxxs,"wys":wys,"wyys":wyys}
dir = os.getcwd()+"/data/"
sio.savemat(dir+"nn_data_rcn_kb"+".mat", mdict)















