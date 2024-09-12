import numpy as np
from scipy.fft import fft2, fftfreq, fftshift, ifft2
import os
import sys
import time
import matplotlib.pyplot as plt
from utils import t6hat
import math
import scipy.io as sio
from scipy import special


########################################################################################################################
########################################################################################################################


dirac_factor = 1e-15
R=.5


def freq_grids(xlen,xnum,ylen,ynum):
    """
    makes fourier frequency grids
    """
    kxx = (2. * np.pi / xlen) * fftfreq(xnum, 1. / xnum)
    kyy = (2. * np.pi / ylen) * fftfreq(ynum, 1. / ynum)
    return np.meshgrid(kxx, kyy)


# def column_samples(scale,subsampling_factor,xlength):
#     """
#     Returns column indices of middle, subsampled rectangle of a meshgrid where X changes along columns and is centered at 0
#     """
#     return np.where((X[0, :] > -round(scale * xlength / 2))
#                     & (X[0, :] < round(scale * xlength / 2)))[0][::subsampling_factor]
#
#
# def row_samples(scale, subsampling_factor,ylength):
#     """
#     Returns row indices of middle, subsampled rectangle of a meshgrid where Y changes along rows and is centered at 0
#     """
#     return np.where((Y[:,0]>-round(scale*ylength/2)) &
#                     (Y[:,0]<round(scale*ylength/2)))[0][::subsampling_factor]


# def sigma(rmax_x,rmin_x,rmax_y,rmin_y,xshift,yshift):
#     """
#     makes a smooth indicator function
#     """
#     return t6hat(rmax_x, rmin_x, X - xshift) * t6hat(rmax_y, rmin_y, Y - yshift)


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


def DiracDelta(arr):
    return (1./(np.sqrt(np.pi)*np.abs(dirac_factor)))*np.exp(-(arr/dirac_factor)**2)


def d_DiracDelta(arr):
    return (2.*arr*np.exp(-(arr/dirac_factor)**2))/(np.sqrt(np.pi)*dirac_factor**2*np.abs(dirac_factor))


# mu determines sharpness of knee bend


logfile = open(os.getcwd()+"/logs/sh_dislocation/wv_gen_sh_statdisloc_v5.out", 'w')
sys.stdout = logfile


start = time.time()


# option to print derivative terms using sympy
# option to set sign of amplitude
print_grad = True
print_hess = False
amp_pos = True


data = sio.loadmat(os.getcwd()+"/data/sh_dislocation/"+"SH_Disloc_v2_13.mat")
Wfull = data['uu'][:,:,-1]
xxfull = data['xx'].T[0]
yyfull = data['yy'].T[0]
Xfull,Yfull = np.meshgrid(xxfull,yyfull)
nyfull,nxfull = np.shape(Wfull)
Winner = Wfull[int(nyfull/4):int(3*nyfull/4),int(nxfull/4):int(3*nxfull/4)]
Xinner = Xfull[int(nyfull/4):int(3*nyfull/4),int(nxfull/4):int(3*nxfull/4)]
Yinner = Yfull[int(nyfull/4):int(3*nyfull/4),int(nxfull/4):int(3*nxfull/4)]
Winner = Winner.T
nyinner, nxinner = np.shape(Winner)
Lxinner = Xinner[0,-1]-Xinner[0,0]
Lyinner = Yinner[-1,0]-Yinner[0,0]
xiinner, etainner = freq_grids(Lxinner,nxinner,Lyinner,nyinner)


def gaussian(x0,y0,X,Y,sigma):
    """
    gaussian bump
    """
    exponent = (X-x0)**2 + (Y-y0)**2
    return np.exp(-exponent/(sigma**2))

def theta(kb,beta,X,Y):
    return X*kb + 1.0*np.log((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
            special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) +
            0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)/beta



def obj(kb, beta, phi, X, Y, W):
    """
    function to be minimized
    """
    theta =  X*kb + 1.0*np.log((0.5 - 0.5*np.exp(np.pi*beta*np.sign(X)))*
            special.erf(Y*np.sqrt(beta*kb)/np.sqrt(np.abs(X))) +
            0.5*np.exp(np.pi*beta*np.sign(X)) + 0.5)/beta
    dtheta_dx = theta_x(kb,beta,X,Y)
    dtheta_dy = theta_y(kb,beta,X,Y)
    k_sq = dtheta_dx**2 + dtheta_dy**2
    if amp_pos:
        amp1 = np.sqrt((4./3.)*(R-(k_sq-1)**2))
    else:
        amp1 = -np.sqrt((4. / 3.) * (R - (k_sq - 1) ** 2))
    amp3 = (amp1 ** 3) / (4 * (R - (9 * k_sq - 1) ** 2))
    return np.mean((amp1 * np.cos(theta - phi) + amp3 * np.cos(3 * (theta - phi)) - W) ** 2)


# def dodkb_string():
#     return "(4 (-w+1/Sqrt[3] 2 Cos[phi-kb x-Log[0.5 +0.5 E^(b \[Pi])+0.5 E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]]/b] \[Sqrt](0.5 -(-1+(1.27324 E^(2 b (\[Pi]-(kb y^2)/x)) kb)/(b x (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2)+(kb-(0.56419 E^(b (\[Pi]-(kb y^2)/x)) kb y)/(Sqrt[b kb] x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])))^2)^2)) (-Cos[phi-kb x-Log[0.5 +0.5 E^(b \[Pi])+0.5 E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]]/b] (-1+(1.27324 E^(2 b (\[Pi]-(kb y^2)/x)) kb)/(b x (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2)+(kb-(0.56419 E^(b (\[Pi]-(kb y^2)/x)) kb y)/(Sqrt[b kb] x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])))^2) (-((1.4367 E^(3 b (\[Pi]-(kb y^2)/x)) kb y)/(Sqrt[b kb] x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^3))+(1.27324 E^(2 b (\[Pi]-(kb y^2)/x)))/(b x (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2)-(2.54648 E^(2 b (\[Pi]-(kb y^2)/x)) kb y^2)/(x^2 (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2)+2 (kb-(0.56419 E^(b (\[Pi]-(kb y^2)/x)) kb y)/(Sqrt[b kb] x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]))) (1+(0.31831 E^(2 b (\[Pi]-(kb y^2)/x)) y^2)/(x^2 (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2)+(0.56419 E^(b (\[Pi]-(kb y^2)/x)) (-0.5 x y+1. b kb y^3))/(Sqrt[b kb] x^(5/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]))))-(-x-(0.56419 E^(b (\[Pi]-(kb y^2)/x)) y)/(Sqrt[b kb] Sqrt[x] (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]))) (0.5 -(-1+(1.27324 E^(2 b (\[Pi]-(kb y^2)/x)) kb)/(b x (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2)+(kb-(0.56419 E^(b (\[Pi]-(kb y^2)/x)) kb y)/(Sqrt[b kb] x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])))^2)^2) Sin[phi-kb x-Log[0.5 +0.5 E^(b \[Pi])+0.5 E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]]/b]))/(Sqrt[3] \[Sqrt](0.5 -(-1+(1.27324 E^(2 b (\[Pi]-(kb y^2)/x)) kb)/(b x (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2)+(kb-(0.56419 E^(b (\[Pi]-(kb y^2)/x)) kb y)/(Sqrt[b kb] x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])))^2)^2))"
#
# def dodb_string():
#     return "(2.3094 (-w+1/Sqrt[3] 2 Cos[phi-kb x-Log[0.5 +0.5 E^(b \[Pi])+0.5 E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]]/b] \[Sqrt](0.5 -(-1+(1.27324 E^(2 b (\[Pi]-(kb y^2)/x)) kb)/(b x (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2)+(kb-(0.56419 E^(b (\[Pi]-(kb y^2)/x)) kb y)/(Sqrt[b kb] x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])))^2)^2)) (1. b^2 Cos[phi-kb x-Log[0.5 +0.5 E^(b \[Pi])+0.5 E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]]/b] (0.159155 E^(2 b (\[Pi]-(kb y^2)/x)) kb Sqrt[b kb] x^3 (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])+0.31831 E^(2 b (\[Pi]-(kb y^2)/x)) (b kb)^(3/2) x^2 (-3.14159 x+kb y^2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])+0.179587 b E^(3 b (\[Pi]-(kb y^2)/x)) kb x^(5/2) (5.56833 E^((b kb y^2)/x) Sqrt[b kb] Sqrt[x]+1. kb y+5.56833 E^((b kb y^2)/x) Sqrt[b kb] Sqrt[x] Erf[(Sqrt[b kb] y)/Sqrt[x]])-0.141047 E^(b (\[Pi]-(3 kb y^2)/x)) kb y (b (1. E^((b kb y^2)/x)+1. E^(b (\[Pi]+(kb y^2)/x))) kb x^(3/2)-0.56419 E^(b \[Pi]) Sqrt[b kb] y+1. b E^(b (\[Pi]+(kb y^2)/x)) kb x^(3/2) Erf[(Sqrt[b kb] y)/Sqrt[x]]) (0.56419 E^(b \[Pi]) Sqrt[b kb] Sqrt[x] y+E^(b (\[Pi]+(kb y^2)/x)) (0.5 x+1. b kb y^2)+E^((b kb y^2)/x) (0.5 x-3.14159 b x+1. b kb y^2)+E^(b (\[Pi]+(kb y^2)/x)) (0.5 x+1. b kb y^2) Erf[(Sqrt[b kb] y)/Sqrt[x]])) (0.31831 b E^(2 b (\[Pi]-(kb y^2)/x)) kb x^2-0.25 b^2 x^3 (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2+(0.282095 E^(b (\[Pi]-(kb y^2)/x)) Sqrt[b kb] y-0.5 b kb x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]))^2)+0.5 Sqrt[x] (0.0625 b^4 x^6 (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^4-2. (0.31831 b E^(2 b (\[Pi]-(kb y^2)/x)) kb x^2-0.25 b^2 x^3 (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2+(0.282095 E^(b (\[Pi]-(kb y^2)/x)) Sqrt[b kb] y-0.5 b kb x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]))^2)^2) (b E^(b \[Pi]) (1.5708 Sqrt[b kb] Sqrt[x]+0.282095 E^(-((b kb y^2)/x)) kb y+1.5708 Sqrt[b kb] Sqrt[x] Erf[(Sqrt[b kb] y)/Sqrt[x]])-0.5 Sqrt[b kb] Sqrt[x] (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]) Log[0.5 +0.5 E^(b \[Pi])+0.5 E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]]) Sin[phi-kb x-Log[0.5 +0.5 E^(b \[Pi])+0.5 E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]]/b]))/(b^6 Sqrt[b kb] x^7 (0.5 +0.5 E^(b \[Pi])+0.5 E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^5 \[Sqrt](0.5 -(-1+(1.27324 E^(2 b (\[Pi]-(kb y^2)/x)) kb)/(b x (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2)+(kb-(0.56419 E^(b (\[Pi]-(kb y^2)/x)) kb y)/(Sqrt[b kb] x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])))^2)^2))"
#
# def dodphi_string():
#     return "-(1/(3 Sqrt[3]))4 \[Sqrt](0.5 -(-1+(1.27324 E^(2 b (\[Pi]-(kb y^2)/x)) kb)/(b x (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2)+(kb-(0.56419 E^(b (\[Pi]-(kb y^2)/x)) kb y)/(Sqrt[b kb] x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])))^2)^2) (-3 w+2 Sqrt[3] Cos[phi-kb x-Log[0.5 +0.5 E^(b \[Pi])+0.5 E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]]/b] \[Sqrt](0.5 -(-1+(1.27324 E^(2 b (\[Pi]-(kb y^2)/x)) kb)/(b x (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])^2)+(kb-(0.56419 E^(b (\[Pi]-(kb y^2)/x)) kb y)/(Sqrt[b kb] x^(3/2) (1. +1. E^(b \[Pi])+1. E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]])))^2)^2)) Sin[phi-kb x-Log[0.5 +0.5 E^(b \[Pi])+0.5 E^(b \[Pi]) Erf[(Sqrt[b kb] y)/Sqrt[x]]]/b]"


diff = 1e-5
def grad_obj_fd(kb,b,phi,X,Y,W):
    """
    gradient of objective function
    """
    do_db = np.mean(
        (obj(kb, b+diff, phi, X, Y, W) - obj(kb, b, phi, X, Y, W)) / diff
    )
    do_dphi = np.mean(
        (obj(kb, b, phi + diff, X, Y, W) - obj(kb, b, phi, X, Y, W)) / diff
    )
    return np.array([do_db,do_dphi])

# get initial estimate of wave number, beta, phase shift
g = gaussian(Xinner[0, int(nxinner/4)], Yinner[int(nyinner/2), 0], Xinner, Yinner, 3.3)
f = g*Winner
spec = fftshift(fft2(f))
max_spec_idx = np.argsort(-np.abs(spec).flatten())[0]
kx0 = np.abs(fftshift(xiinner).flatten()[max_spec_idx])
ky0 = np.abs(fftshift(etainner).flatten()[max_spec_idx])
kb0 = 1.00
beta0 = .49*kb0
phi0 = 0

#do the opimization for x>0 with cutoff
cutoff = 100
Xhalf = Xinner[:,int(nxinner/2)+cutoff:]
Yhalf = Yinner[:,int(nxinner/2)+cutoff:]
Whalf = Winner[:,int(nxinner/2)+cutoff:]
initial_theta = theta(kb0,beta0,Xhalf,Yhalf)-phi0
initial_dtheta_dx = theta_x(kb0,beta0,Xhalf,Yhalf)
initial_dtheta_dy = theta_y(kb0,beta0,Xhalf,Yhalf)
initial_k_sq = initial_dtheta_dx**2 + initial_dtheta_dy**2
if amp_pos:
    initial_amp1 = np.sqrt((4. / 3.) * (R - (initial_k_sq - 1) ** 2))
else:
    initial_amp1 = -np.sqrt((4. / 3.) * (R - (initial_k_sq - 1) ** 2))
initial_pattern = initial_amp1*np.cos(initial_theta)

fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(Whalf)
im1 = axs[1].imshow(initial_pattern)
im2 = axs[2].imshow(np.abs(Whalf-initial_pattern))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Pattern, Approx Pattern, and Error")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_dislocation/FieldEstInit_v1_0508.png")
print("Init Field max err:", np.max(np.abs(Whalf-initial_pattern)))
print("Init Field mean err:", np.mean(np.abs(Whalf-initial_pattern)))

#perform gradient descent on objective function, MSE((A1*cos(phase(k11,k12,k21,k22))-W)^2)
step = .001
max_its = 10000
i = 0
print("Init Vals:",kb0, beta0, phi0)
while np.linalg.norm(grad_obj_fd(kb0,beta0,phi0,Xhalf,Yhalf,Whalf))>1e-4 and i < max_its:
    curr = np.array([beta0,phi0])
    grad = grad_obj_fd(kb0,curr[0],curr[1],Xhalf,Yhalf,Whalf)
    d = step
    new = curr - d*grad
    while obj(kb0,new[0],new[1],Xhalf,Yhalf,Whalf)>obj(kb0,curr[0],curr[1],Xhalf,Yhalf,Whalf):
        print("Objective increased, decreasing step size")
        d*=.5
        new = curr - d * grad
        if d<1e-16:
            print("Norm of step size excessively small")
            print("Step: ", i)
            print("Gradient Norm", np.linalg.norm(grad))
            print("Obj Function Norm: ", np.linalg.norm(obj(kb0,beta0,phi0,Xhalf,Yhalf,Whalf)))
            print("New Vals: ", kb0, beta0, phi0)
            break
    beta0, phi0 = new
    i += 1
    print("Step: ", i)
    print("Gradient Norm", np.linalg.norm(grad))
    print("Obj Function Norm: ",np.linalg.norm(obj(kb0,beta0,phi0,Xhalf,Yhalf,Whalf)))
    print("New Vals: ", kb0, beta0, phi0)

final_theta = theta(kb0,beta0,Xhalf,Yhalf)-phi0
final_dtheta_dx = theta_x(kb0,beta0,Xhalf,Yhalf)
final_dtheta_dy = theta_y(kb0,beta0,Xhalf,Yhalf)
final_k_sq = final_dtheta_dx**2 + final_dtheta_dy**2
if amp_pos:
    final_amp1 = np.sqrt((4. / 3.) * (R - (final_k_sq - 1) ** 2))
else:
    final_amp1 = -np.sqrt((4. / 3.) * (R - (final_k_sq - 1) ** 2))
final_pattern = final_amp1*np.cos(final_theta)
fig, axs = plt.subplots(nrows=1,ncols=3)
im0 = axs[0].imshow(Whalf)
im1 = axs[1].imshow(final_pattern)
im2 = axs[2].imshow(np.abs(Whalf-final_pattern))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Pattern, Approx Pattern, and Error")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_dislocation/FieldEst_v1_0508.png")
print("Est Field max err:", np.max(np.abs(Whalf-final_pattern)))
print("Est Field mean err:", np.mean(np.abs(Whalf-final_pattern)))


logfile.close()


