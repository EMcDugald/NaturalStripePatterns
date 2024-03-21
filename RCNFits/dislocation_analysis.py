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
start = time.time()

# set up geometry and parameters for pattern
Lx = 20*np.pi
Ly = Lx
Nx = 256
Ny = 256
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


def curlk(kb, beta,Xhalf,Yhalf):
    """
    curl of wave vector
    """
    return theta_yx(kb,beta,Xhalf,Yhalf)-\
           theta_xy(kb,beta,Xhalf,Yhalf)

def Jk(kb,beta,Xhalf,Yhalf):
    """
    jacobian determinant of wave vector (aka, hessian determinant of phase)
    """
    return theta_xx(kb,beta,Xhalf,Yhalf)*theta_yy(kb,beta,Xhalf,Yhalf)-\
           theta_xy(kb,beta,Xhalf,Yhalf)*theta_yx(kb,beta,Xhalf,Yhalf)


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


### NEW STUFF HERE ###
def gaussian(x0,y0,X,Y,sigma):
    """
    gaussian bump
    """
    exponent = (X-x0)**2 + (Y-y0)**2
    return np.exp(-exponent/(sigma**2))

def obj(kb, beta, X, Y, W):
    """
    function to be minimized
    """
    theta =  X*kb + 1.0*np.log((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
            special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) +
            0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)/beta
    return np.mean((np.cos(theta)-W)**2)

def grad_obj(kb, beta, X, Y, W):
    """
    gradient of objective function
    """
    do_dkb = np.mean(
        -2 * (-W + np.cos(X * kb + 1.0 * np.log((0.5 - 0.5 * np.exp(np.pi * beta * np.sign(X))) * special.erf(
            Y * np.sqrt(beta * kb) / np.sqrt(np.abs(X))) + 0.5 * np.exp(np.pi * beta * np.sign(X)) + 0.5) / beta)) * (
                    X + 1.0 * Y * np.sqrt(beta * kb) * (0.5 - 0.5 * np.exp(np.pi * beta * np.sign(X))) * np.exp(
                -Y ** 2 * beta * kb / np.abs(X)) / (np.sqrt(np.pi) * beta * kb * (
                        (0.5 - 0.5 * np.exp(np.pi * beta * np.sign(X))) * special.erf(
                    Y * np.sqrt(beta * kb) / np.sqrt(np.abs(X))) + 0.5 * np.exp(
                    np.pi * beta * np.sign(X)) + 0.5) * np.sqrt(np.abs(X)))) * np.sin(X * kb + 1.0 * np.log(
            (0.5 - 0.5 * np.exp(np.pi * beta * np.sign(X))) * special.erf(
                Y * np.sqrt(beta * kb) / np.sqrt(np.abs(X))) + 0.5 * np.exp(np.pi * beta * np.sign(X)) + 0.5) / beta)
    )
    do_dbeta = np.mean(
        -2 * (-W + np.cos(X * kb + 1.0 * np.log((0.5 - 0.5 * np.exp(np.pi * beta * np.sign(X))) * special.erf(
            Y * np.sqrt(beta * kb) / np.sqrt(np.abs(X))) + 0.5 * np.exp(np.pi * beta * np.sign(X)) + 0.5) / beta)) * (
                    1.0 * (Y * np.sqrt(beta * kb) * (0.5 - 0.5 * np.exp(np.pi * beta * np.sign(X))) * np.exp(
                -Y ** 2 * beta * kb / np.abs(X)) / (np.sqrt(np.pi) * beta * np.sqrt(np.abs(X))) - 0.5 * np.pi * np.exp(
                np.pi * beta * np.sign(X)) * special.erf(Y * np.sqrt(beta * kb) / np.sqrt(np.abs(X))) * np.sign(
                X) + 0.5 * np.pi * np.exp(np.pi * beta * np.sign(X)) * np.sign(X)) / (beta * (
                        (0.5 - 0.5 * np.exp(np.pi * beta * np.sign(X))) * special.erf(
                    Y * np.sqrt(beta * kb) / np.sqrt(np.abs(X))) + 0.5 * np.exp(
                    np.pi * beta * np.sign(X)) + 0.5)) - 1.0 * np.log(
                (0.5 - 0.5 * np.exp(np.pi * beta * np.sign(X))) * special.erf(
                    Y * np.sqrt(beta * kb) / np.sqrt(np.abs(X))) + 0.5 * np.exp(
                    np.pi * beta * np.sign(X)) + 0.5) / beta ** 2) * np.sin(X * kb + 1.0 * np.log(
            (0.5 - 0.5 * np.exp(np.pi * beta * np.sign(X))) * special.erf(
                Y * np.sqrt(beta * kb) / np.sqrt(np.abs(X))) + 0.5 * np.exp(np.pi * beta * np.sign(X)) + 0.5) / beta)

    )
    return np.array([do_dkb, do_dbeta])


def freq_grids(xlen,xnum,ylen,ynum):
    """
    makes fourier frequency grids
    """
    kxx = (2. * np.pi / xlen) * fftfreq(xnum, 1. / xnum)
    kyy = (2. * np.pi / ylen) * fftfreq(Ny, 1. / ynum)
    return np.meshgrid(kxx, kyy)



# # compute exact phase, exact pattern, and exact phase gradient
kb_exact = 1.0
#beta_exact = .01 #originally was .01
beta_exact = .5

theta_exact_half = theta(kb_exact,beta_exact,Xhalf,Yhalf)
theta_x_exact_half = theta_x(kb_exact,beta_exact, Xhalf, Yhalf)
theta_y_exact_half = theta_y(kb_exact,beta_exact, Xhalf, Yhalf)

theta_exact_full = np.zeros(shape=(Ny,Nx))
theta_exact_full[:,int(Nx/2):] += theta_exact_half
theta_exact_full[:,0:int(Nx/2)] += np.flip(theta_exact_half,1)
W = np.cos(theta_exact_full)
Whalf = W[:,int(Nx/2):]

theta_x_exact_full = np.zeros(shape=(Ny,Nx))
theta_x_exact_full[:,int(Nx/2):] += theta_x_exact_half
theta_x_exact_full[:,0:int(Nx/2)] -= np.flip(theta_x_exact_half,1) #derivative is odd in x

theta_y_exact_full = np.zeros(shape=(Ny,Nx))
theta_y_exact_full[:,int(Nx/2):] += theta_y_exact_half
theta_y_exact_full[:,0:int(Nx/2)] += np.flip(theta_y_exact_half,1)


# make frequency grid
xi, eta = freq_grids(Lx,Nx,Ly,Ny)


# get initial estimate of wave number in left half plane
g = gaussian(X[0, int(Nx/4)], Y[int(Ny/2), 0], X, Y, 3.3)
f = g*W
spec = fftshift(fft2(f))
max_spec_idx = np.argsort(-np.abs(spec).flatten())[0]
kx0 = np.abs(fftshift(xi).flatten()[max_spec_idx])
ky0 = np.abs(fftshift(eta).flatten()[max_spec_idx])
kb0 = np.sqrt(kx0**2+ky0**2)
beta0 = .6*kb0


# optionally, use sympy to get derivatives of the objective function
if print_grad:
    import sympy as sp
    kbsym,betasym,xsym,ysym,wsym = sp.symbols('kb,beta,X,Y,W', real=True)
    phase_fn_sym = kbsym*xsym + (1./betasym)*sp.log(
        .5*(1+sp.exp(betasym*sp.pi*sp.sign(xsym))) +
        .5*(1-sp.exp(betasym*sp.pi*sp.sign(xsym)))*sp.erf(sp.sqrt(betasym*kbsym)*ysym/sp.sqrt(sp.Abs(xsym)))
    )
    obj_fn_sym = (sp.cos(phase_fn_sym)-wsym)**2
    print("Phase Function:", "\n", str(phase_fn_sym).replace("log","np.log").
          replace("exp","np.exp").replace("erf","special.erf").replace("sign","np.sign").
          replace("Abs","np.abs").replace("pi","np.pi").replace("sqrt","np.sqrt").replace("sin","np.sin").replace("cos","np.cos"))
    print("Objective Function:", "\n", str(obj_fn_sym).replace("log","np.log").
          replace("exp","np.exp").replace("erf","special.erf").replace("sign","np.sign").
          replace("Abs","np.abs").replace("pi","np.pi").replace("sqrt","np.sqrt").replace("sin","np.sin").replace("cos","np.cos"))
    print("Objective Function Partial in kb:", "\n", str(obj_fn_sym.diff(kbsym)).replace("log","np.log").
          replace("exp","np.exp").replace("erf","special.erf").replace("sign","np.sign").
          replace("Abs","np.abs").replace("pi","np.pi").replace("sqrt","np.sqrt").replace("sin","np.sin").replace("cos","np.cos"))
    print("Objective Function Partial in beta:", "\n",  str(obj_fn_sym.diff(betasym)).replace("log","np.log").
          replace("exp","np.exp").replace("erf","special.erf").replace("sign","np.sign").
          replace("Abs","np.abs").replace("pi","np.pi").replace("sqrt","np.sqrt").replace("sin","np.sin").replace("cos","np.cos"))
    print("Phase Function Partial in x:", "\n", str(phase_fn_sym.diff(xsym)).replace("log","np.log").
          replace("exp","np.exp").replace("erf","special.erf").replace("sign","np.sign").
          replace("Abs","np.abs").replace("pi","np.pi").replace("sqrt","np.sqrt").replace("sin","np.sin").replace("cos","np.cos"))
    print("Phase Function Partial in y:", "\n", str(phase_fn_sym.diff(ysym)).replace("log","np.log").
          replace("exp","np.exp").replace("erf","special.erf").replace("sign","np.sign").
          replace("Abs","np.abs").replace("pi","np.pi").replace("sqrt","np.sqrt").replace("sin","np.sin").replace("cos","np.cos"))
    print("Phase Function Partial in xx:", "\n", str(phase_fn_sym.diff(xsym).diff(xsym)).replace("log","np.log").
          replace("exp","np.exp").replace("erf","special.erf").replace("sign","np.sign").
          replace("Abs","np.abs").replace("pi","np.pi").replace("sqrt","np.sqrt").replace("sin","np.sin").replace("cos","np.cos"))
    print("Phase Function Partial in yy:", "\n", str(phase_fn_sym.diff(ysym).diff(ysym)).replace("log","np.log").
          replace("exp","np.exp").replace("erf","special.erf").replace("sign","np.sign").
          replace("Abs","np.abs").replace("pi","np.pi").replace("sqrt","np.sqrt").replace("sin","np.sin").replace("cos","np.cos"))
    print("Phase Function Partial in xy:", "\n", str(phase_fn_sym.diff(xsym).diff(ysym)).replace("log","np.log").
          replace("exp","np.exp").replace("erf","special.erf").replace("sign","np.sign").
          replace("Abs","np.abs").replace("pi","np.pi").replace("sqrt","np.sqrt").replace("sin","np.sin").replace("cos","np.cos"))
    print("Phase Function Partial in yx:", "\n", str(phase_fn_sym.diff(ysym).diff(xsym)).replace("log","np.log").
          replace("exp","np.exp").replace("erf","special.erf").replace("sign","np.sign").
          replace("Abs","np.abs").replace("pi","np.pi").replace("sqrt","np.sqrt").replace("sin","np.sin").replace("cos","np.cos"))



# perform gradient descent on objective function, MSE(cos(phase(kb,beta))-W)^2)
#step = .001
step = .01
max_its = 20000
i = 0
print("Init Vals:",kb0,beta0)
while np.linalg.norm(grad_obj(kb0,beta0,Xhalf,Yhalf,Whalf))>1e-6 and i < max_its:
    curr = np.array([kb0,beta0])
    grad = grad_obj(curr[0],curr[1],Xhalf,Yhalf,Whalf)
    d = step
    new = curr - d*grad
    while obj(new[0],new[1],Xhalf,Yhalf,Whalf)>obj(curr[0],curr[1],Xhalf,Yhalf,Whalf):
        print("Objective increased, decreasing step size")
        d*=.5
        new = curr - d * grad
        if d<1e-16:
            print("Norm of step size excessively small")
            print("Step: ", i)
            print("Gradient Norm", np.linalg.norm(grad))
            print("Obj Function Norm: ", np.linalg.norm(obj(kb0,beta0,Xhalf,Yhalf,Whalf)))
            print("New Vals: ",kb0,beta0)
            break
    kb0, beta0 = new
    i += 1
    print("Step: ", i)
    print("Gradient Norm", np.linalg.norm(grad))
    print("Obj Function Norm: ",np.linalg.norm(obj(kb0,beta0,Xhalf,Yhalf,Whalf)))
    print("New Vals: ", kb0,beta0)

print("exact kb:",kb_exact)
print("exact beta:",beta_exact)
print("found kb:",kb0)
print("found beta:",beta0)

#compare recovered pattern to given data
final_theta_half = theta(kb0,beta0,Xhalf,Yhalf)
final_theta_full = np.zeros(shape=(Ny,Nx))
final_theta_full[:,int(Nx/2):] += final_theta_half
final_theta_full[:,0:int(Nx/2)] += np.flip(final_theta_half,1)

final_pattern = np.cos(final_theta_full)
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(W)
im1 = axs[1].imshow(final_pattern)
im2 = axs[2].imshow(np.abs(W-final_pattern))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Pattern, Approx Pattern, and Error")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/FieldEst_v5.png")
print("Est Field max err:", np.max(np.abs(W-final_pattern)))
print("Est Field mean err:", np.mean(np.abs(W-final_pattern)))


fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(theta_exact_full)
im1 = axs[1].imshow(final_theta_full)
im2 = axs[2].imshow(np.abs(theta_exact_full-final_theta_full))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Phase, Approx Phase, and Error")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/PhaseEst_v5.png")
print("Est phase max err:", np.max(np.abs(theta_exact_full-final_theta_full)))
print("Est phase mean err:", np.mean(np.abs(theta_exact_full-final_theta_full)))

s = np.sign(X)
F_exact = np.exp(beta_exact*s*theta_exact_full - beta_exact*kb_exact*X)
F_approx = np.exp(beta0*s*final_theta_full - beta0*kb0*X)
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(F_exact)
im1 = axs[1].imshow(F_approx)
im2 = axs[2].imshow(np.abs(F_exact-F_approx))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Exact vs Approx F")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/F_v5.png")
print("F max err:", np.max(np.abs(F_exact-F_approx)))
print("F mean err:", np.mean(np.abs(F_exact-F_approx)))



# compute derivatives on each half independently
gap = .1*Lx
x1 = -Lx/2+gap/2
x2 = -gap/2
x3 = gap/2
x4 = Lx/2-gap/2
y1 = -Ly/2+gap/2
y2 = Ly/2-gap/2
dx = xx[1]-xx[0]
dy = yy[1]-yy[0]

dfdx = np.zeros(shape=np.shape(theta_exact_full))
dfdy = np.zeros(shape=np.shape(theta_exact_full))
dfdxx = np.zeros(shape=np.shape(theta_exact_full))
dfdyy = np.zeros(shape=np.shape(theta_exact_full))
dfdxy = np.zeros(shape=np.shape(theta_exact_full))
dfdyx = np.zeros(shape=np.shape(theta_exact_full))
Fx = np.zeros(shape=np.shape(theta_exact_full))
Fyy = np.zeros(shape=np.shape(theta_exact_full))

m,n = np.shape(theta_exact_full)
dfdx[2:-2,2:-2] = FiniteDiffDerivs4(final_theta_full,dx,dy,type='x')
dfdy[2:-2,2:-2] = FiniteDiffDerivs4(final_theta_full,dx,dy,type='y')
dfdxx[2:-2,2:-2] = FiniteDiffDerivs4(final_theta_full,dx,dy,type='xx')
dfdyy[2:-2,2:-2] = FiniteDiffDerivs4(final_theta_full,dx,dy,type='yy')
dfdxy[4:-4,4:-4] = FiniteDiffDerivs4(final_theta_full,dx,dy,type='xy')
dfdyx[4:-4,4:-4] = FiniteDiffDerivs4(final_theta_full,dx,dy,type='yx')
Fx[2:-2,2:-2] = FiniteDiffDerivs4(F_approx,dx,dy,type='x')
Fyy[2:-2,2:-2] = FiniteDiffDerivs4(F_approx,dx,dy,type='yy')

# use sliding gaussian /smooth indicator window and ffts to get partial derivatives of phase
deriv_cols = np.where(((X[::ss_factor,::ss_factor][0, :] > x1) & (X[::ss_factor,::ss_factor][0,:] < x2)) |
                ((X[::ss_factor,::ss_factor][0,:]>x3) & (X[::ss_factor,::ss_factor][0,:]<x4)))[0]
deriv_rows = np.where((Y[::ss_factor,::ss_factor][:,0]>y1) & (Y[::ss_factor,::ss_factor][:,0]<y2))[0]

rows_full, cols_full = np.indices(np.shape(X))
rows_ss = rows_full[:,0][::ss_factor]
cols_ss = cols_full[0,:][::ss_factor]


theta_x_approx = np.zeros((len(rows_ss),len(cols_ss)))
theta_y_approx = np.zeros((len(rows_ss),len(cols_ss)))
theta_xx_approx = np.zeros((len(rows_ss),len(cols_ss)))
theta_yy_approx = np.zeros((len(rows_ss),len(cols_ss)))
theta_xy_approx = np.zeros((len(rows_ss),len(cols_ss)))
theta_yx_approx = np.zeros((len(rows_ss),len(cols_ss)))
F_x_approx = np.zeros((len(rows_ss),len(cols_ss)))
F_yy_approx = np.zeros((len(rows_ss),len(cols_ss)))

print("Making derivative grids")
start2 = time.time()
for r in rows_ss:
    for c in cols_ss:
        if (r in ss_factor*deriv_rows) and (c in ss_factor*deriv_cols):
            theta_x_approx[int(r/ss_factor),int(c/ss_factor)] += dfdx[r, c]
            theta_y_approx[int(r/ss_factor),int(c/ss_factor)] += dfdy[r, c]
            theta_xx_approx[int(r/ss_factor),int(c/ss_factor)] += dfdxx[r, c]
            theta_yy_approx[int(r/ss_factor),int(c/ss_factor)] += dfdyy[r, c]
            theta_xy_approx[int(r/ss_factor),int(c/ss_factor)] += dfdxy[r, c]
            theta_yx_approx[int(r/ss_factor),int(c/ss_factor)] += dfdyx[r, c]
            F_x_approx[int(r/ss_factor),int(c/ss_factor)] += Fx[r,c]
            F_yy_approx[int(r/ss_factor),int(c/ss_factor)] += Fyy[r,c]
        else:
            theta_x_approx[int(r/ss_factor),int(c/ss_factor)] += np.nan
            theta_x_approx[int(r/ss_factor),int(c/ss_factor)] += np.nan
            theta_y_approx[int(r/ss_factor),int(c/ss_factor)] += np.nan
            theta_xx_approx[int(r/ss_factor),int(c/ss_factor)] += np.nan
            theta_yy_approx[int(r/ss_factor),int(c/ss_factor)] += np.nan
            theta_xy_approx[int(r/ss_factor),int(c/ss_factor)] += np.nan
            theta_yx_approx[int(r/ss_factor),int(c/ss_factor)] += np.nan
            F_x_approx[int(r / ss_factor), int(c / ss_factor)] += np.nan
            F_yy_approx[int(r / ss_factor), int(c / ss_factor)] += np.nan
end2 = time.time()
print("time to make derivatives:",end2-start2)


theta_x_exact_ss = theta_x_exact_full[::ss_factor,::ss_factor]
theta_y_exact_ss = theta_y_exact_full[::ss_factor,::ss_factor]


divk_exact_half = divk(kb_exact,beta_exact,Xhalf,Yhalf)
divk_exact_full = np.zeros(shape=(Ny,Nx))
divk_exact_full[:,int(Nx/2):] += divk_exact_half
divk_exact_full[:,0:int(Nx/2)] += np.flip(divk_exact_half,1)
divk_exact_ss = divk_exact_full[::ss_factor,::ss_factor]

curlk_exact_half = curlk(kb_exact,beta_exact,Xhalf,Yhalf)
curlk_exact_full = np.zeros(shape=(Ny,Nx))
curlk_exact_full[:,int(Nx/2):] += curlk_exact_half
curlk_exact_full[:,0:int(Nx/2)] += np.flip(curlk_exact_half,1)
curlk_exact_ss = curlk_exact_full[::ss_factor,::ss_factor]

Jk_exact_half = Jk(kb_exact,beta_exact,Xhalf,Yhalf)
Jk_exact_full = np.zeros(shape=(Ny,Nx))
Jk_exact_full[:,int(Nx/2):] += curlk_exact_half
Jk_exact_full[:,0:int(Nx/2)] += np.flip(Jk_exact_half,1)
Jk_exact_ss = Jk_exact_full[::ss_factor,::ss_factor]

divk_approx = theta_xx_approx+theta_yy_approx
curlk_approx = theta_yx_approx-theta_xy_approx
Jk_approx = theta_xx_approx*theta_yy_approx - theta_xy_approx*theta_yx_approx

exact_ss_wavenums = np.sqrt(theta_x_exact_ss**2+theta_y_exact_ss**2)
wavenums_approx = np.sqrt(theta_x_approx**2+theta_y_approx**2)

F_approx_ss = F_approx[::ss_factor,::ss_factor]
# F_x_approx_ss = F_x_approx[::ss_factor,::ss_factor]
# F_yy_approx_ss = F_yy_approx[::ss_factor,::ss_factor]


theta_x_exact_ss[np.where(theta_x_approx==np.nan)] = np.nan
theta_y_exact_ss[np.where(theta_y_approx==np.nan)] = np.nan
exact_ss_wavenums[np.where(wavenums_approx==np.nan)] = np.nan
divk_exact_ss[np.where(divk_approx==np.nan)] = np.nan
curlk_exact_ss[np.where(curlk_approx==np.nan)] = np.nan
Jk_exact_ss[np.where(Jk_approx==np.nan)] = np.nan

# F_x_approx[np.where(F_x_approx==np.nan)] = np.nan
# F_yy_approx[np.where(F_yy_approx==np.nan)] = np.nan
F_approx_ss[np.where(np.isnan(F_x_approx))] = np.nan


#compare recovered phase gradient to exact phase gradient
fig, axs = plt.subplots(nrows=2,ncols=2)
im0 = axs[0,0].imshow(theta_x_exact_ss)
im1 = axs[0,1].imshow(theta_y_exact_ss)
im2 = axs[1,0].imshow(theta_x_approx)
im3 = axs[1,1].imshow(theta_y_approx)
im2.set_clim(np.min(theta_x_exact_ss),np.max(theta_x_exact_ss))
im3.set_clim(np.min(theta_y_exact_ss),np.max(theta_y_exact_ss))
plt.colorbar(im0,ax=axs[0,0])
plt.colorbar(im1,ax=axs[0,1])
plt.colorbar(im2,ax=axs[1,0])
plt.colorbar(im3,ax=axs[1,1])
plt.suptitle("Exact vs Approx Phase Gradient")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/PhaseGradients_v5.png")
print("theta_x max err:", np.nanmax(np.abs(theta_x_exact_ss-theta_x_approx)))
print("theta_x mean err:", np.nanmean(np.abs(theta_x_exact_ss-theta_x_approx)))
print("theta_y max err:", np.nanmax(np.abs(theta_y_exact_ss-theta_y_approx)))
print("theta_y mean err:", np.nanmean(np.abs(theta_y_exact_ss-theta_y_approx)))

#compare recovered wave nums to exact wave nums
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(exact_ss_wavenums)
im1 = axs[1].imshow(wavenums_approx)
im2 = axs[2].imshow(np.abs(exact_ss_wavenums-wavenums_approx))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Exact vs Approx Wave Nums")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/WaveNums_v5.png")
print("wave num max err:", np.nanmax(np.abs(exact_ss_wavenums-wavenums_approx)))
print("wave num mean err:", np.nanmean(np.abs(exact_ss_wavenums-wavenums_approx)))

#compare recovered divk to exact divk
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(divk_exact_ss)
im1 = axs[1].imshow(divk_approx)
im2 = axs[2].imshow(np.abs(divk_exact_ss-divk_approx))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Exact vs Approx Div(k)")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/DivK_v5.png")
print("Div(k) max err:", np.nanmax(np.abs(divk_exact_ss-divk_approx)))
print("Div(k) mean err:", np.nanmean(np.abs(divk_exact_ss-divk_approx)))

#compare recovered curl to exact curl
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(curlk_exact_ss)
im1 = axs[1].imshow(curlk_approx)
im2 = axs[2].imshow(np.abs(curlk_exact_ss-curlk_approx))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Exact vs Approx Curl(k)")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/CurlK_v5.png")
print("Curl(k) max err:", np.nanmax(np.abs(curlk_exact_ss-curlk_approx)))
print("Curl(k) mean err:", np.nanmean(np.abs(curlk_exact_ss-curlk_approx)))

#compare recovered jacobian to exact jacobian
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(Jk_exact_ss)
im1 = axs[1].imshow(Jk_approx)
im2 = axs[2].imshow(np.abs(Jk_exact_ss-Jk_approx))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Exact vs Approx J(k)")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/Jk_v5.png")
print("J(k) max err:", np.nanmax(np.abs(Jk_exact_ss-Jk_approx)))
print("J(k) mean err:", np.nanmean(np.abs(Jk_exact_ss-Jk_approx)))
print("Gap:",gap)


end = time.time()
print("Total Time:",end-start)


print("F Function PDE Mean Abs Err:", np.nanmean(np.abs(2*beta0*kb0*F_x_approx+F_yy_approx)))
print("F Function PDE Max Abs Err:", np.nanmax(np.abs(2*beta0*kb0*F_x_approx+F_yy_approx)))

# save data
mdict = {'theta_x_approx':theta_x_approx,'theta_y_approx': theta_y_approx,
         'theta_xx_approx': theta_xx_approx, 'theta_yy_approx': theta_yy_approx,
         'theta_xy_approx':theta_xy_approx,'theta_yx_approx':theta_yx_approx,
         'divk_approx': divk_approx,'curlk_approx': curlk_approx,'Jk_approx':Jk_approx,
         'wavenums_approx':wavenums_approx, 'F_x_approx':F_x_approx,'F_yy_approx':F_yy_approx,
         'F_approx':F_approx_ss, 'kb':kb0,'beta':beta0}

sio.savemat(os.getcwd()+"/data/dislocation/v5.mat",mdict)


print("Test Standard SD L2:",np.linalg.norm(divk_approx[~np.isnan(divk_approx)]**2-1+2*wavenums_approx[~np.isnan(wavenums_approx)]-wavenums_approx[~np.isnan(wavenums_approx)]**2))
print("Test Standard SD Mean:",np.mean(np.abs(divk_approx[~np.isnan(divk_approx)]**2-1+2*wavenums_approx[~np.isnan(wavenums_approx)]-wavenums_approx[~np.isnan(wavenums_approx)]**2)))
print("Test Standard SD Max:",np.max(np.abs(divk_approx[~np.isnan(divk_approx)]**2-1+2*wavenums_approx[~np.isnan(wavenums_approx)]-wavenums_approx[~np.isnan(wavenums_approx)]**2)))

print("Test F Equation L2 Right:",np.linalg.norm(2*beta*kb*F_x_approx[~np.isnan(F_x_approx)]+F_yy_approx[~np.isnan(F_yy_approx)]))
print("Test F Equation Mean Right:",np.mean(np.abs(2*beta*kb*F_x_approx[~np.isnan(F_x_approx)]+F_yy_approx[~np.isnan(F_yy_approx)])))
print("Test F Equation Max Right:",np.max(np.abs(2*beta*kb*F_x_approx[~np.isnan(F_x_approx)]+F_yy_approx[~np.isnan(F_yy_approx)])))
