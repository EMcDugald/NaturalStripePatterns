import numpy as np
from scipy.fft import fft2, fftfreq, fftshift
import os
import matplotlib.pyplot as plt
import sys
import time
import scipy.io as sio
from scipy import special
from derivatives import FiniteDiffDerivs4

# option to print derivative terms using sympy
print_grad = True

# set up geometry and parameters for pattern
Lx = 20*np.pi
Ly = Lx
Nx = 512
Ny = 512
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
xx = np.arange(-Lx/2,Lx/2+dx/2,dx)
yy = np.arange(-Ly/2,Ly/2+dy/2,dy)
X,Y = np.meshgrid(xx,yy)
ss_factor = 2
dirac_factor = 1e-15

print("Grid Dims:", "Nx = ",Nx, "Ny = ",Ny)
print("Dom Size:", "Lx = ",Lx, "Ly = ", Ly)
print("Approximation subsampling:", ss_factor)

xx_half = xx[int(Nx/2):]
Xhalf,Yhalf = np.meshgrid(xx_half,yy)


def DiracDelta(arr):
    return (1./(np.sqrt(np.pi)*np.abs(dirac_factor)))*np.exp(-(arr/dirac_factor)**2)

def d_DiracDelta(arr):
    return (2.*arr*np.exp(-(arr/dirac_factor)**2))/(np.sqrt(np.pi)*dirac_factor**2*np.abs(dirac_factor))

def theta(kb,beta,X,Y):
    """
    phase
    """
    return X*kb + 1.0*np.log((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
                             special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) +
                             0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)/beta

def theta_x(kb,beta,X,Y):
    """
    partial derivative in x of phase
    """

    return kb + 1.0*(-Y*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
                     np.exp(-Y**2*beta*kb/np.abs(X))*np.sign(X)/(np.sqrt(np.pi)*np.abs(X)**(3/2))
                     - 1.0*np.pi*beta*np.exp(np.pi*beta*np.sign(X))*DiracDelta(X)*
                     special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) +
                     1.0*np.pi*beta*np.exp(np.pi*beta*np.sign(X))*DiracDelta(X))/\
           (beta*((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*special.erf(Y*np.sqrt(beta*kb)/
            np.sqrt(np.abs(X))) + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5))

def theta_y(kb,beta,X,Y):
    """
    partial derivative in y of phase
    """
    return 2.0*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*\
           np.exp(-Y**2*beta*kb/np.abs(X))/(np.sqrt(np.pi)*beta*
        ((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X)))
         + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)*np.sqrt(np.abs(X)))

def theta_xx(kb,beta,X,Y):
    """
    second derivative of phase in x
    """
    return 1.0*(2.0*np.sqrt(np.pi)*Y*beta*np.sqrt(beta*kb)*np.exp(np.pi*beta*np.sign(X))*
                np.exp(-Y**2*beta*kb/np.abs(X))*DiracDelta(X)*np.sign(X)/np.abs(X)**(3/2) -
                2*Y*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
                np.exp(-Y**2*beta*kb/np.abs(X))*DiracDelta(X)/(np.sqrt(np.pi)*np.abs(X)**(3/2)) +
                3*Y*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
                np.exp(-Y**2*beta*kb/np.abs(X))*np.sign(X)**2/(2*np.sqrt(np.pi)*np.abs(X)**(5/2)) -
                2.0*np.pi**2*beta**2*np.exp(np.pi*beta*np.sign(X))*DiracDelta(X)**2*
                special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) + 2.0*np.pi**2*beta**2*
                np.exp(np.pi*beta*np.sign(X))*DiracDelta(X)**2 - 1.0*np.pi*beta*
                np.exp(np.pi*beta*np.sign(X))*d_DiracDelta(X)*special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) +
                1.0*np.pi*beta*np.exp(np.pi*beta*np.sign(X))*d_DiracDelta(X) - Y**3*beta*kb*np.sqrt(beta*kb)*(0.5 -
                0.5*np.exp(np.pi*beta*np.sign(X)))*np.exp(-Y**2*beta*kb/np.abs(X))*np.sign(X)**2/
                (np.sqrt(np.pi)*X**2*np.abs(X)**(3/2)))/(beta*((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
                special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)) + \
                1.0*(-Y*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*np.exp(-Y**2*beta*kb/np.abs(X))*
                np.sign(X)/(np.sqrt(np.pi)*np.abs(X)**(3/2)) - 1.0*np.pi*beta*np.exp(np.pi*beta*np.sign(X))*
                DiracDelta(X)*special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) + 1.0*np.pi*beta*
                np.exp(np.pi*beta*np.sign(X))*DiracDelta(X))*(Y*np.sqrt(beta*kb)*(0.5 -
                0.5*np.exp(np.pi*beta*np.sign(X)))*np.exp(-Y**2*beta*kb/np.abs(X))*np.sign(X)
                /(np.sqrt(np.pi)*np.abs(X)**(3/2)) + 1.0*np.pi*beta*np.exp(np.pi*beta*np.sign(X))*
                DiracDelta(X)*special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) -
                1.0*np.pi*beta*np.exp(np.pi*beta*np.sign(X))*DiracDelta(X))/(beta*((0.5 -
               0.5*np.exp(np.pi*beta*np.sign(X)))*special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) +
                0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)**2)

def theta_yy(kb,beta,X,Y):
    """
    second derivative of phase in y
    """
    return -4.0*Y*kb*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*\
           np.exp(-Y**2*beta*kb/np.abs(X))/(np.sqrt(np.pi)*((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
        special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)*
        np.abs(X)**(3/2)) - 4.0*beta*kb*0.25*(1 - np.exp(np.pi*beta*np.sign(X)))**2*np.exp(-2*Y**2*beta*kb
    /np.abs(X))/(np.pi*beta*((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*special.erf(Y*np.sqrt(beta*kb)
    /np.sqrt(np.abs(X))) + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)**2*np.abs(X))

def theta_xy(kb,beta,X,Y):
    """
    derivative of phase in xy
    """
    return -2.0*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*\
           (-Y*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
            np.exp(-Y**2*beta*kb/np.abs(X))*np.sign(X)/(np.sqrt(np.pi)*np.abs(X)**(3/2))
            - 1.0*np.pi*beta*np.exp(np.pi*beta*np.sign(X))*DiracDelta(X)*special.erf(Y*np.sqrt(beta*kb)
        /np.sqrt(np.abs(X))) + 1.0*np.pi*beta*np.exp(np.pi*beta*np.sign(X))*DiracDelta(X))*\
    np.exp(-Y**2*beta*kb/np.abs(X))/(np.sqrt(np.pi)*beta*((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
    special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)**2*
    np.sqrt(np.abs(X))) + 1.0*(2*Y**2*beta*kb*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
    np.exp(-Y**2*beta*kb/np.abs(X))*np.sign(X)/(np.sqrt(np.pi)*np.abs(X)**(5/2)) -
    2.0*np.sqrt(np.pi)*beta*np.sqrt(beta*kb)*np.exp(np.pi*beta*np.sign(X))*np.exp(-Y**2*beta*kb/np.abs(X))*
    DiracDelta(X)/np.sqrt(np.abs(X)) - np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
    np.exp(-Y**2*beta*kb/np.abs(X))*np.sign(X)/(np.sqrt(np.pi)*np.abs(X)**(3/2)))/\
    (beta*((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*special.erf(Y*np.sqrt(beta*kb)/
        np.sqrt(np.abs(X))) + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5))

def theta_yx(kb,beta,X,Y):
    """
    derivative of phase in yx
    """
    return -2.0*np.sqrt(np.pi)*np.sqrt(beta*kb)*np.exp(np.pi*beta*np.sign(X))*np.exp(-Y**2*beta*kb/np.abs(X))*\
           DiracDelta(X)/(((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*special.erf(Y*np.sqrt(beta*kb)/
        np.sqrt(np.abs(X))) + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)*np.sqrt(np.abs(X))) - \
        1.0*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*np.exp(-Y**2*beta*kb/np.abs(X))*\
        np.sign(X)/(np.sqrt(np.pi)*beta*((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*special.erf(Y*np.sqrt(beta*kb)/
        np.sqrt(np.abs(X))) + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)*np.abs(X)**(3/2)) + \
        2.0*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*(Y*np.sqrt(beta*kb)*(0.5 -
        0.5*np.exp(np.pi*beta*np.sign(X)))*np.exp(-Y**2*beta*kb/np.abs(X))*np.sign(X)/(np.sqrt(np.pi)*
        np.abs(X)**(3/2)) + 1.0*np.pi*beta*np.exp(np.pi*beta*np.sign(X))*DiracDelta(X)*
        special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) - 1.0*np.pi*beta*np.exp(np.pi*beta*np.sign(X))*
        DiracDelta(X))*np.exp(-Y**2*beta*kb/np.abs(X))/(np.sqrt(np.pi)*beta*((0.5 - 0.5*
        np.exp(np.pi*beta*np.sign(X)))*special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) +
        0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)**2*np.sqrt(np.abs(X))) + 2.0*Y**2*kb*np.sqrt(beta*kb)*(0.5 -
        0.5*np.exp(np.pi*beta*np.sign(X)))*np.exp(-Y**2*beta*kb/np.abs(X))*np.sign(X)/\
        (np.sqrt(np.pi)*X**2*((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*special.erf(Y*np.sqrt(beta*kb)/
        np.sqrt(np.abs(X))) + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)*np.sqrt(np.abs(X)))


def divk(kb, beta,Xhalf,Yhalf):
    """
    divergence of wave vector (aka, laplacian of phase)
    """
    return theta_xx(kb,beta,Xhalf,Yhalf)+\
           theta_yy(kb,beta,Xhalf,Yhalf)


kb = 1.0
beta = .5

theta_half = theta(kb,beta,Xhalf,Yhalf)
theta_full = np.zeros(shape=(Ny,Nx))
theta_full[:,int(Nx/2):] += theta_half
theta_full[:,0:int(Nx/2)] += np.flip(theta_half,1)

pattern = np.cos(theta_full)

theta_x_half = theta_x(kb,beta,Xhalf,Yhalf)
theta_x_full = np.zeros(shape=(Ny,Nx))
theta_x_full[:,int(Nx/2):] += theta_x_half
theta_x_full[:,0:int(Nx/2)] += np.flip(theta_x_half,1)
print("Max theta_x:", np.max(theta_x_full))
print("Min theta_x:", np.min(theta_x_full))

theta_y_half = theta_y(kb,beta,Xhalf,Yhalf)
theta_y_full = np.zeros(shape=(Ny,Nx))
theta_y_full[:,int(Nx/2):] += theta_y_half
theta_y_full[:,0:int(Nx/2)] += np.flip(theta_y_half,1)
print("Max theta_y:", np.max(theta_y_full))
print("Min theta_y:", np.min(theta_y_full))
wavenum = np.sqrt(theta_x_full**2+theta_y_full**2)

divk_half = divk(kb,beta,Xhalf,Yhalf)
divk_full = np.zeros(shape=(Ny,Nx))
divk_full[:,int(Nx/2):] += divk_half
divk_full[:,0:int(Nx/2)] += np.flip(divk_half,1)

fig, axs = plt.subplots(nrows=1,ncols=2)
im0 = axs[0].imshow(theta_full,cmap='bwr')
im1 = axs[1].imshow(pattern,cmap='bwr')
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Tst phase, pattern")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/analysis/PhasePatt.png")

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(theta_x_full,cmap='bwr')
im1 = axs[1].imshow(theta_y_full,cmap='bwr')
im2 = axs[2].imshow(wavenum,cmap='bwr',clim=[0,2])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Tst phase gradient, wave number")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/analysis/GradWN.png")

print("Max divk:", np.max(divk_full))
print("Min divk:", np.min(divk_full))

F_right = np.exp(beta*theta_half - beta*kb*Xhalf)
F_full = np.zeros(shape=(Ny,Nx))
F_full[:,int(Nx/2):] += F_right
F_full[:,0:int(Nx/2)] += np.flip(F_right,1)
fig, ax = plt.subplots(nrows=1,ncols=1)
im0 = ax.imshow(theta_full,cmap='bwr')
plt.colorbar(im0,ax=axs[0])
plt.suptitle("F field")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/analysis/F.png")

F_x_right = beta*np.exp(beta*(theta_half-kb*Xhalf))*(theta_x_half-kb)
F_yy_right = beta**2*np.exp(beta*(theta_half-kb*Xhalf))*theta_y_half**2 + \
             beta*np.exp(beta*(theta_half-kb*Xhalf))*theta_yy(kb,beta,Xhalf,Yhalf)

print("Test Standard SD L2:",np.linalg.norm(divk_full**2-1+2*wavenum-wavenum**2))
print("Test Standard SD Mean:",np.mean(np.abs(divk_full**2-1+2*wavenum-wavenum**2)))
print("Test Standard SD Max:",np.max(np.abs(divk_full**2-1+2*wavenum-wavenum**2)))

print("Test Modified SD L2:",np.linalg.norm(divk_full**2-1./4. - (1./4.)*wavenum**2 - (1./32.)*wavenum**3))
print("Test Modified SD Mean:",np.mean(np.abs(divk_full**2-1./4. - (1./4.)*wavenum**2 - (1./32.)*wavenum**3)))
print("Test Modified SD Max:",np.max(np.abs(divk_full**2-1./4. - (1./4.)*wavenum**2 - (1./32.)*wavenum**3)))

print("Test F Equation L2:",np.linalg.norm(2*beta*kb*F_x_right+F_yy_right))
print("Test F Equation Mean:",np.mean(np.abs(2*beta*kb*F_x_right+F_yy_right)))
print("Test F Equation Max:",np.max(np.abs(2*beta*kb*F_x_right+F_yy_right)))

print("Truncating middle column")


print("Test Standard SD L2 Left:",np.linalg.norm(divk_full[:,:-int(Nx/2)-40]**2-1+2*wavenum[:,:-int(Nx/2)-40]-wavenum[:,:-int(Nx/2)-40]**2))
print("Test Standard SD Mean Left:",np.mean(np.abs(divk_full[:,:-int(Nx/2)-40]**2-1+2*wavenum[:,:-int(Nx/2)-40]-wavenum[:,:-int(Nx/2)-40]**2)))
print("Test Standard SD Max Left:",np.max(np.abs(divk_full[:,:-int(Nx/2)-40]**2-1+2*wavenum[:,:-int(Nx/2)-40]-wavenum[:,:-int(Nx/2)-40]**2)))

print("Test Standard SD L2 Right:",np.linalg.norm(divk_full[:,int(Nx/2)+40:int(Nx)]**2-1+2*wavenum[:,int(Nx/2)+40:int(Nx)]-wavenum[:,int(Nx/2)+40:int(Nx)]**2))
print("Test Standard SD Mean Right:",np.mean(np.abs(divk_full[:,int(Nx/2)+40:int(Nx)]**2-1+2*wavenum[:,int(Nx/2)+40:int(Nx)]-wavenum[:,int(Nx/2)+40:int(Nx)]**2)))
print("Test Standard SD Max Right:",np.max(np.abs(divk_full[:,int(Nx/2)+40:int(Nx)]**2-1+2*wavenum[:,int(Nx/2)+40:int(Nx)]-wavenum[:,int(Nx/2)+40:int(Nx)]**2)))

print("Test Modified SD L2 Right:",np.linalg.norm(divk_full[:,int(Nx/2)+40:int(Nx)]**2-1./4. - (1./4.)*wavenum[:,int(Nx/2)+40:int(Nx)]**2 - (1./32.)*wavenum[:,int(Nx/2)+40:int(Nx)]**3))
print("Test Modified SD Mean:",np.mean(np.abs(divk_full[:,int(Nx/2)+40:int(Nx)]**2-1./4. - (1./4.)*wavenum[:,int(Nx/2)+40:int(Nx)]**2 - (1./32.)*wavenum[:,int(Nx/2)+40:int(Nx)]**3)))
print("Test Modified SD Max:",np.max(np.abs(divk_full[:,int(Nx/2)+40:int(Nx)]**2-1./4. - (1./4.)*wavenum[:,int(Nx/2)+40:int(Nx)]**2 - (1./32.)*wavenum[:,int(Nx/2)+40:int(Nx)]**3)))

print("Test F Equation L2 Right:",np.linalg.norm(2*beta*kb*F_x_right[:,40:]+F_yy_right[:,40:]))
print("Test F Equation Mean Right:",np.mean(np.abs(2*beta*kb*F_x_right[:,40:]+F_yy_right[:,40:])))
print("Test F Equation Max Right:",np.max(np.abs(2*beta*kb*F_x_right[:,40:]+F_yy_right[:,40:])))
