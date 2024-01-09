import numpy as np
from scipy.fft import fft2, fftfreq, fftshift, ifft2
import os
import matplotlib.pyplot as plt
import sys
import time
from utils import t6hat
import math
import scipy.io as sio
from scipy import special

#########################################################################################################################
    ### This script compute wave vector fields from stationary dislocation using the RCN knee bend ansatz ###
    ### We assume the phase is of the form a*ln(e^(k11*x + k12*y) + e^(k21*x + k22*y)) ###
    ### Step 1: Multiply the field, W,  by a narrow Gaussian in upper half plane ###
    ### Step 2: Take FFT of the result and use frequency corresponding to dominant mode for initial wave vector guess ###
    ### Step 3: Use this guess and the RCN ansatz to compute an initial phase surface, cos(phase) ###
    ### Step 4: Use gradient descent on MSE((cos(phase)-W)^2), with phase = phase(a,k11,k12,k21,k22) ###
    ### Step 5: Use gaussian window to compute spectral partial derivatives of the recovered phase ###
    ### Step 6: Perform a SINDy optimization procedure on wave vectors, fit to self dual solution ###
    ### Note: The optimization can be done with a as a free parameter as well. See back.txt for an example ###
    ### Note: The file kb_scratch.py demonstrates more derivative methods ###
#########################################################################################################################



# logfile = open(os.getcwd()+"/logs/dislocation/wv_gen_dislocation_v1.out", 'w')
# sys.stdout = logfile

start = time.time()

# option to print derivative terms using sympy
print_grad = True

# set up geometry and parameters for pattern
Lx = 30*np.pi
Ly = Lx
Nx = 512
Ny = 512
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
xx = np.arange(-Lx/2,Lx/2+dx/2,dx)
yy = np.arange(-Ly/2,Ly/2+dy/2,dy)
X,Y = np.meshgrid(xx,yy)
ss_factor = 4
len_scale = .75
dirac_factor = 1e-12

print("Grid Dims:", "Nx = ",Nx, "Ny = ",Ny)
print("Dom Size:", "Lx = ",Lx, "Ly = ", Ly)
print("Approximation length scale:", len_scale)
print("Approximation subsampling:", ss_factor)


def DiracDelta(arr):
    return (1./(np.sqrt(np.pi)*np.abs(dirac_factor)))*np.exp(-(arr/dirac_factor)**2)

def d_DiracDelta(arr):
    return (2.*arr*np.exp(-(arr/dirac_factor)**2))/(np.sqrt(np.pi)*dirac_factor**2*np.abs(dirac_factor))

def theta(kb,beta):
    """
    phase
    """
    return X*kb + 1.0*np.log((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
                             special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) +
                             0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)/beta


def theta_x(kb,beta):
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

def theta_y(kb,beta):
    """
    partial derivative in y of phase
    """
    return 2.0*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*\
           np.exp(-Y**2*beta*kb/np.abs(X))/(np.sqrt(np.pi)*beta*
        ((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X)))
         + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)*np.sqrt(np.abs(X)))

def theta_xx(kb,beta):
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

def theta_yy(kb,beta):
    """
    second derivative of phase in y
    """
    return -4.0*Y*kb*np.sqrt(beta*kb)*(0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*\
           np.exp(-Y**2*beta*kb/np.abs(X))/(np.sqrt(np.pi)*((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
        special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)*
        np.abs(X)**(3/2)) - 4.0*beta*kb*0.25*(1 - np.exp(np.pi*beta*np.sign(X)))**2*np.exp(-2*Y**2*beta*kb
    /np.abs(X))/(np.pi*beta*((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*special.erf(Y*np.sqrt(beta*kb)
    /np.sqrt(np.abs(X))) + 0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)**2*np.abs(X))

def theta_xy(kb,beta):
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

def theta_yx(kb,beta):
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


def divk(kb, beta):
    """
    divergence of wave vector (aka, laplacian of phase)
    """
    return theta_xx(kb,beta)+theta_yy(kb,beta)

def curlk(kb, beta):
    """
    curl of wave vector
    """
    return theta_yx(kb,beta)-theta_xy(kb,beta)

def Jk(kb,beta):
    """
    jacobian determinant of wave vector (aka, hessian determinant of phase)
    """
    return theta_xx(kb,beta)*theta_yy(kb,beta)-theta_xy(kb,beta)*theta_yx(kb,beta)





kb_tst = 1.0
beta_tst = .01

tst_theta = theta(kb_tst,beta_tst)
tst_pattern = np.cos(tst_theta)
tst_theta_x = theta_x(kb_tst,beta_tst)
print("Max tst_theta_x:", np.max(tst_theta_x))
print("Min tst_theta_x:", np.min(tst_theta_x))
tst_theta_y = theta_y(kb_tst,beta_tst)
print("Max tst_theta_y:", np.max(tst_theta_y))
print("Min tst_theta_y:", np.min(tst_theta_y))
tst_wavenum = np.sqrt(tst_theta_x**2+tst_theta_y**2)
tst_divk = divk(kb_tst,beta_tst)
tst_curlk = curlk(kb_tst,beta_tst)
tst_Jk = Jk(kb_tst,beta_tst)

fig, axs = plt.subplots(nrows=1,ncols=2)
im0 = axs[0].imshow(tst_theta,cmap='bwr')
im1 = axs[1].imshow(tst_pattern,cmap='bwr')
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Tst phase, pattern")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/TstPhasePatt_v1.png")

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(tst_theta_x,cmap='bwr')
im1 = axs[1].imshow(tst_theta_y,cmap='bwr')
im2 = axs[2].imshow(tst_wavenum,cmap='bwr',clim=[0,2])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Tst phase gradient, wave number")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/TstGradWN_v1.png")

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(tst_divk,cmap='bwr')
im1 = axs[1].imshow(tst_curlk,cmap='bwr')
im2 = axs[2].imshow(tst_Jk,cmap='bwr')
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Tst divk, curlk, Jk")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/TstDivCurlJ_v1.png")

print("Max tst_divk:", np.max(tst_divk))
print("Min tst_divk:", np.min(tst_divk))
print("Max tst_curlk:", np.max(tst_curlk))
print("Min tst_curlk:", np.min(tst_curlk))
print("Max tst_Jk:", np.max(tst_Jk))
print("Min tst_Jk:", np.min(tst_Jk))

W = tst_pattern

def gaussian(x0,y0,X,Y,sigma):
    """
    gaussian bump
    """
    exponent = (X-x0)**2 + (Y-y0)**2
    return np.exp(-exponent/(sigma**2))

def obj(kb, beta):
    """
    function to be minimized
    """
    theta =  X*kb + 1.0*np.log((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
            special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) +
            0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)/beta
    return np.mean((np.cos(theta)-W)**2)

def grad_obj(kb, beta):
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

def column_samples(scale,subsampling_factor,xlength):
    """
    Returns column indices of middle, subsampled rectangle of a meshgrid where X changes along columns and is centered at 0
    """
    return np.where((X[0, :] > -round(scale * xlength / 2))
                    & (X[0, :] < round(scale * xlength / 2)))[0][::subsampling_factor]

def row_samples(scale, subsampling_factor,ylength):
    """
    Returns row indices of middle, subsampled rectangle of a meshgrid where Y changes along rows and is centered at 0
    """
    return np.where((Y[:,0]>-round(scale*ylength/2)) &
                    (Y[:,0]<round(scale*ylength/2)))[0][::subsampling_factor]

def freq_grids(xlen,xnum,ylen,ynum):
    """
    makes fourier frequency grids
    """
    kxx = (2. * np.pi / xlen) * fftfreq(xnum, 1. / xnum)
    kyy = (2. * np.pi / ylen) * fftfreq(Ny, 1. / ynum)
    return np.meshgrid(kxx, kyy)

def sigma(rmax_x,rmin_x,rmax_y,rmin_y,xshift,yshift):
    """
    makes a smooth indicator function
    """
    return t6hat(rmax_x, rmin_x, X - xshift) * t6hat(rmax_y, rmin_y, Y - yshift)



# # compute exact phase, exact pattern, and exact phase gradient
kb_exact = 1.0
beta_exact = .01
theta_exact = theta(kb_exact,beta_exact)
W = np.cos(theta_exact)
theta_x_exact = theta_x(kb_exact,beta_exact)
theta_y_exact = theta_y(kb_exact,beta_exact)


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
beta0 = .25*kb0


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
step = .01
max_its = 5000
i = 0
print("Init Vals:",kb0,beta0)
while np.linalg.norm(grad_obj(kb0,beta0))>1e-10 and i < max_its:
    curr = np.array([kb0,beta0])
    grad = grad_obj(curr[0],curr[1])
    d = step
    new = curr - d*grad
    while obj(new[0],new[1])>obj(curr[0],curr[1]):
        print("Objective increased, decreasing step size")
        d*=.5
        new = curr - d * grad
        if d<1e-16:
            print("Norm of step size excessively small")
            print("Step: ", i)
            print("Gradient Norm", np.linalg.norm(grad))
            print("Obj Function Norm: ", np.linalg.norm(obj(kb0,beta0)))
            print("New Vals: ",kb0,beta0)
            break
    kb0, beta0 = new
    i += 1
    print("Step: ", i)
    print("Gradient Norm", np.linalg.norm(grad))
    print("Obj Function Norm: ",np.linalg.norm(obj(kb0,beta0)))
    print("New Vals: ", kb0,beta0)

print("exact kb:",kb_exact)
print("exact beta:",beta_exact)
print("found kb:",kb0)
print("found beta:",beta0)

#compare recovered pattern to given data
final_theta = theta(kb0,beta0)
final_pattern = np.cos(final_theta)
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(W)
im1 = axs[1].imshow(final_pattern)
im2 = axs[2].imshow(np.abs(W-final_pattern))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Pattern, Approx Pattern, and Error")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/FieldEst_v1.png")
print("Est Field max err:", np.max(np.abs(W-final_pattern)))
print("Est Field mean err:", np.mean(np.abs(W-final_pattern)))


# use sliding gaussian /smooth indicator window and ffts to get partial derivatives of phase
cols = column_samples(len_scale,ss_factor,Lx)
rows = row_samples(len_scale,ss_factor,Ly)
theta_x_approx = np.zeros((len(rows),len(cols)))
theta_y_approx = np.zeros((len(rows),len(cols)))
theta_xx_approx = np.zeros((len(rows),len(cols)))
theta_yy_approx = np.zeros((len(rows),len(cols)))
theta_xy_approx = np.zeros((len(rows),len(cols)))
theta_yx_approx = np.zeros((len(rows),len(cols)))
print("Shape of subsampled interior grid:",np.shape(theta_x_approx))
r_shift = rows[0]
c_shift = cols[0]
dx = xx[1]-xx[0]
dy = yy[1]-yy[0]
rmax_x = .95*(math.floor((1-len_scale)*Lx/2)-dx)
rmax_y = .95*(math.floor((1-len_scale)*Ly/2)-dy)
print("rmax_x:",rmax_x,"rmax_y:",rmax_y)
print("Making derivative grids")
start2 = time.time()
for r in rows:
    for c in cols:
        s = sigma(rmax_x, .05*rmax_x, rmax_y, .05*rmax_y, X[r, c], Y[r, c])
        f = s*final_theta
        dfdx = np.real(ifft2(1j*xi*fft2(f)))
        theta_x_approx[round((r - r_shift)/ss_factor), round((c - c_shift)/ss_factor)] += dfdx[r,c]
        dfdy = np.real(ifft2(1j*eta*fft2(f)))
        theta_y_approx[round((r - r_shift)/ss_factor), round((c - c_shift)/ss_factor)] += dfdy[r, c]
        dfdxx = np.real(ifft2((1j*xi)**2*fft2(f)))
        theta_xx_approx[round((r - r_shift) / ss_factor), round((c - c_shift) / ss_factor)] += dfdxx[r, c]
        dfdyy = np.real(ifft2((1j * eta) ** 2 * fft2(f)))
        theta_yy_approx[round((r - r_shift) / ss_factor), round((c - c_shift) / ss_factor)] += dfdyy[r, c]
        dfdxy = np.real(ifft2((1j*eta)*(1j * xi) * fft2(f)))
        theta_xy_approx[round((r - r_shift) / ss_factor), round((c - c_shift) / ss_factor)] += dfdxy[r, c]
        dfdyx = np.real(ifft2((1j * xi) *(1j * eta) * fft2(f)))
        theta_yx_approx[round((r - r_shift) / ss_factor), round((c - c_shift) / ss_factor)] += dfdyx[r, c]
end2 = time.time()
print("time to make derivatives:",end2-start2)


theta_x_exact_ss = theta_x_exact[rows[0]:rows[-1]+ss_factor,cols[0]:cols[-1]+ss_factor][::ss_factor,::ss_factor]
theta_y_exact_ss = theta_y_exact[rows[0]:rows[-1]+ss_factor,cols[0]:cols[-1]+ss_factor][::ss_factor,::ss_factor]
divk_exact = divk(kb_exact,beta_exact)
divk_exact_ss = divk_exact[rows[0]:rows[-1]+ss_factor,cols[0]:cols[-1]+ss_factor][::ss_factor,::ss_factor]
curlk_exact = curlk(kb_exact,beta_exact)
curlk_exact_ss = curlk_exact[rows[0]:rows[-1]+ss_factor,cols[0]:cols[-1]+ss_factor][::ss_factor,::ss_factor]
Jk_exact = Jk(kb_exact,beta_exact)
Jk_exact_ss = Jk_exact[rows[0]:rows[-1]+ss_factor,cols[0]:cols[-1]+ss_factor][::ss_factor,::ss_factor]
divk_approx = theta_xx_approx+theta_yy_approx
curlk_approx = theta_yx_approx-theta_xy_approx
Jk_approx = theta_xx_approx*theta_yy_approx - theta_xy_approx*theta_yx_approx

# get coordinates cooresponding to derivative arrays
Xinterior = X[rows[0]:rows[-1]+ss_factor,cols[0]:cols[-1]+ss_factor][::ss_factor,::ss_factor]
Yinterior = Y[rows[0]:rows[-1]+ss_factor,cols[0]:cols[-1]+ss_factor][::ss_factor,::ss_factor]
lenXint = Xinterior[0,:][-1]-Xinterior[0,:][0]
lenYint = Yinterior[:,0][-1]-Yinterior[:,0][0]
print("Intended Lengths of interior grid:", "xlength=",len_scale*Lx, "ylength=",len_scale*Ly)
print("Actual Lengths of interior grid:", "xlength=",lenXint,"ylength=",lenYint)


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
plt.savefig(os.getcwd()+"/figs/dislocation/PhaseGradients_v1.png")
print("theta_x max err:", np.max(np.abs(theta_x_exact_ss-theta_x_approx)))
print("theta_x mean err:", np.mean(np.abs(theta_x_exact_ss-theta_x_approx)))
print("theta_y max err:", np.max(np.abs(theta_y_exact_ss-theta_y_approx)))
print("theta_y mean err:", np.mean(np.abs(theta_y_exact_ss-theta_y_approx)))

#compare recovered wave nums to exact wave nums
exact_ss_wavenums = np.sqrt(theta_x_exact_ss**2+theta_y_exact_ss**2)
wavenums_approx = np.sqrt(theta_x_approx**2+theta_y_approx**2)
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(exact_ss_wavenums)
im1 = axs[1].imshow(wavenums_approx)
im2 = axs[2].imshow(np.abs(exact_ss_wavenums-wavenums_approx))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Exact vs Approx Wave Nums")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/dislocation/WaveNums_v1.png")
print("wave num max err:", np.max(np.abs(exact_ss_wavenums-wavenums_approx)))
print("wave num mean err:", np.mean(np.abs(exact_ss_wavenums-wavenums_approx)))

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
plt.savefig(os.getcwd()+"/figs/dislocation/DivK_v1.png")
print("Div(k) max err:", np.max(np.abs(divk_exact_ss-divk_approx)))
print("Div(k) mean err:", np.mean(np.abs(divk_exact_ss-divk_approx)))

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
plt.savefig(os.getcwd()+"/figs/dislocation/CurlK_v1.png")
print("Curl(k) max err:", np.max(np.abs(curlk_exact_ss-curlk_approx)))
print("Curl(k) mean err:", np.mean(np.abs(curlk_exact_ss-curlk_approx)))

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
plt.savefig(os.getcwd()+"/figs/dislocation/Jk_v1.png")
print("J(k) max err:", np.max(np.abs(Jk_exact_ss-Jk_approx)))
print("J(k) mean err:", np.mean(np.abs(Jk_exact_ss-Jk_approx)))


end = time.time()
print("Total Time:",end-start)


# save data
mdict = {'theta_x_approx':theta_x_approx,'theta_y_approx': theta_y_approx,
         'theta_xx_approx': theta_xx_approx, 'theta_yy_approx': theta_yy_approx,
         'theta_xy_approx':theta_xy_approx,'theta_yx_approx':theta_yx_approx,
         'divk_approx': divk_approx,'curlk_approx': curlk_approx,'Jk_approx':Jk_approx,
         'wavenums_approx':wavenums_approx,'Xinterior':Xinterior,'Yinterior':Yinterior}

sio.savemat(os.getcwd()+"/data/dislocation/v1.mat",mdict)
# logfile.close()

