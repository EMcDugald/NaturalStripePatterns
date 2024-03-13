import numpy as np
from scipy.fft import fft2, fftfreq, fftshift
import os
import matplotlib.pyplot as plt
import sys
import time
import scipy.io as sio
from scipy import special
from derivatives import FiniteDiffDerivs4

Lx = 20*np.pi
Ly = Lx
Nx = 4096
Ny = 4096
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
xx = np.arange(-Lx/2,Lx/2+dx/2,dx)
yy = np.arange(-Ly/2,Ly/2+dy/2,dy)
X,Y = np.meshgrid(xx,yy)
ss_factor = 2
dirac_factor = 1e-15

def theta(kb,beta):
    """
    phase
    """
    return X*kb + (1.0/beta)*np.log(.5*(1+ np.exp(np.pi*beta*np.sign(X))) +
                                    .5*(1-np.exp(np.pi*beta*np.sign(X)))*special.erf(np.sqrt(beta*kb)*Y/np.sqrt(np.abs(X))))

def erf2(x):
    return (2/np.sqrt(np.pi))*(x-x**3/3+x**5/(5*2*1)-x**7/(7*3*2*1))

def theta2(kb,beta):
    """
    phase
    """
    return X*kb + 1.0*np.log((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
                             erf2(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X+1e-8))) +
                             0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)/beta


kb = 1.0
beta = .01

tst_theta = theta(kb,beta)
tst_pattern = np.cos(tst_theta)

fig, axs = plt.subplots(nrows=1,ncols=2)
im0 = axs[0].imshow(tst_theta,cmap='bwr')
im1 = axs[1].imshow(tst_pattern,cmap='bwr')
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Phase, Pattern, kb = {}, beta = {}".format(kb,beta))
plt.tight_layout()
plt.show()

kb = 1.0
beta = 1.0

tst_theta = theta(kb,beta)
tst_pattern = np.cos(tst_theta)

fig, axs = plt.subplots(nrows=1,ncols=2)
im0 = axs[0].imshow(tst_theta,cmap='bwr')
im1 = axs[1].imshow(tst_pattern,cmap='bwr')
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Phase, Pattern, kb = {}, beta = {}".format(kb,beta))
plt.tight_layout()
plt.show()