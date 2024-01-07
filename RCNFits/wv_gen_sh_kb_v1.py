import numpy as np
from scipy.fft import fft2, fftfreq, fftshift, ifft2
import os
import sys
import time
from solve_sh import solve_sh_zigzag
import matplotlib.pyplot as plt

# ToDo: Adding phase shift term
#########################################################################################################################
    ### This script compute wave vector fields from Swift-Hohenberg knee bends using the RCN knee bend ansatz ###
    ### This version just uses an ansatz A1cos(theta), with A1 constant (and possibly treats theta as theta(a,k1,k2))
#########################################################################################################################


def get_centered_pattern(surf,x,y):
    """
    SH solver for knee bends uses an akward domain. This method retrieves a symmetric PGB and normalizes the coordinates
    """
    Ny_full = np.shape(surf)[0]
    W = surf[int(Ny_full / 4) - int(Nx / 2):int(Ny_full / 4) + int(Nx / 2), :]
    Xtmp = x[int(Ny_full / 4) - int(Nx / 2):int(Ny_full / 4) + int(Nx / 2), :]
    Ytmp = y[int(Ny_full / 4) - int(Nx / 2):int(Ny_full / 4) + int(Nx / 2), :]
    xx = Xtmp[0, :]
    yy = Ytmp[:, 0]
    xshift = (xx[int(Nx / 2) - 1] + xx[int(Nx / 2)]) / 2
    yshift = (yy[int(Nx / 2) - 1] + yy[int(Nx / 2)]) / 2
    xxx = xx - xshift
    yyy = yy - yshift
    X, Y = np.meshgrid(xxx, yyy)
    return W, X, Y

def freq_grids(xlen,xnum,ylen,ynum):
    """
    makes fourier frequency grids
    """
    kxx = (2. * np.pi / xlen) * fftfreq(xnum, 1. / xnum)
    kyy = (2. * np.pi / ylen) * fftfreq(Ny, 1. / ynum)
    return np.meshgrid(kxx, kyy)


# mu determines sharpness of knee bend
mu = .3

logfile = open(os.getcwd()+"/logs/sh_pgbs/wv_gen_sh_kb_mu_{}.out".format(mu), 'w')
sys.stdout = logfile

start = time.time()

# option to print derivative terms using sympy
print_grad = True
print_hess = False

# set params for SH Solver
Nx = 256
tmax = 100
R = .5
h = .5

Wfull,Xfull,Yfull = solve_sh_zigzag(Nx,mu,tmax,R,h)
W,X,Y = get_centered_pattern(Wfull,Xfull,Yfull)

xx = X[0,:]
yy = Y[:,0]
Nx = len(xx)
Ny = len(yy)
Lx = xx[-1]-xx[0]
Ly = yy[-1]-yy[0]
ss_factor = 2
len_scale = .7
xi, eta = freq_grids(Lx,Nx,Ly,Ny)
print("Grid Dims:", "Nx = ",Nx, "Ny = ",Ny)
print("Dom Size:", "Lx = ",Lx, "Ly = ", Ly)
print("Approximation length scale:", len_scale)
print("Approximation subsampling:", ss_factor)


def gaussian(x0,y0,X,Y,sigma):
    """
    gaussian bump
    """
    exponent = (X-x0)**2 + (Y-y0)**2
    return np.exp(-exponent/(sigma**2))

def obj(a,A1,k11,k12,k21,k22,phi):
    """
    function to be minimized
    """
    theta = a*np.log(np.exp(k11*X+k12*Y)+np.exp(k21*X+k22*Y))
    return np.mean((A1*np.cos(theta-phi)-W)**2)

def grad_obj(a,A1,k11,k12,k21,k22,phi):
    """
    gradient of objective function
    """
    do_da = np.mean(
        -2*A1*(A1*np.cos(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi) - W)*
        np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22))*
        np.sin(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi)
    )
    do_dA1 = np.mean(
        2*(A1*np.cos(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi) - W)*
        np.cos(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi)
    )
    do_dk11 = np.mean(
        -2*A1*X*a*(A1*np.cos(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi) - W)
        *np.exp(X*k11 + Y*k12)*np.sin(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi)
        /(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22))
    )
    do_dk12 = np.mean(
        -2*A1*Y*a*(A1*np.cos(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi) - W)*
        np.exp(X*k11 + Y*k12)*np.sin(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi)
        /(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22))
    )
    do_dk21 = np.mean(
        -2*A1*X*a*(A1*np.cos(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi) - W)*
        np.exp(X*k21 + Y*k22)*np.sin(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi)/
        (np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22))
    )
    do_dk22 = np.mean(
        -2*A1*Y*a*(A1*np.cos(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi) - W)*
        np.exp(X*k21 + Y*k22)*np.sin(a*np.log(np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22)) - phi)/
        (np.exp(X*k11 + Y*k12) + np.exp(X*k21 + Y*k22))
    )
    do_dphi = np.mean(
        2 * A1 * (A1 * np.cos(a * np.log(np.exp(X * k11 + Y * k12) + np.exp(X * k21 + Y * k22)) - phi) - W)
        * np.sin(a * np.log(np.exp(X * k11 + Y * k12) + np.exp(X * k21 + Y * k22)) - phi)
    )
    return np.array([do_da,do_dA1, do_dk11, do_dk12, do_dk21, do_dk22, do_dphi])

# get initial estimate of wave vector in upper half plane.
# this determines k11, k12 in the RCN phase ansatz, and by symmetry also determines k21 and k22
# multiple upper half plane by gaussian, take fft, find frequency corresponding to dominant mode
g = gaussian(X[0, int(Nx/2)], Y[int(Ny/4), 0], X, Y, 3.3)
f = g*W
spec = fftshift(fft2(f))
max_spec_idx = np.argsort(-np.abs(spec).flatten())[0]
kx0 = np.abs(fftshift(xi).flatten()[max_spec_idx])
ky0 = np.abs(fftshift(eta).flatten()[max_spec_idx])
k110 = kx0
k120 = ky0
k210 = kx0
k220 = -ky0
A10 = np.max(np.abs(W))
print("Max abs W:",np.max(np.abs(W)))
a0 = 1.0
phi0 = np.pi/4


# optionally, use sympy to get derivatives of the objective function
if print_grad:
    import sympy as sp
    asym, A1sym, k11sym, k12sym, k21sym, k22sym, xsym, ysym, wsym, phisym = sp.symbols('a,A1,k11,k12,k21,k22,X,Y,W, phi')
    obj_fn = (A1sym*sp.cos(asym*sp.log(sp.exp(k11sym*xsym + k12sym*ysym)+sp.exp(k21sym*xsym+k22sym*ysym))-phisym)-wsym)**2
    phase_fn = k11sym*xsym + sp.log(2*sp.cosh(k12sym*ysym))
    print("Objective Function:", sp.simplify(obj_fn))
    print("Objective Function Analytical Derivative in a:", sp.simplify(obj_fn.diff(asym)))
    print("Objective Function Analytical Derivative in A1:", sp.simplify(obj_fn.diff(A1sym)))
    print("Objective Function Analytical Derivative in k11:", sp.simplify(obj_fn.diff(k11sym)))
    print("Objecctive Function Analytical Derivative in k12:", sp.simplify(obj_fn.diff(k12sym)))
    print("Objective Function Analytical Derivative in k21:", sp.simplify(obj_fn.diff(k21sym)))
    print("Objective Function Analytical Derivative in k22:", sp.simplify(obj_fn.diff(k22sym)))
    print("Objective Function Analytical Derivative in phi:", sp.simplify(obj_fn.diff(phisym)))
    print("Phase Function all gradient terms:")
    [print(sp.simplify(obj_fn.diff(x))) for x in [asym,A1sym,k11sym,k12sym,k21sym,k22sym,phisym]]
    if print_hess:
        print("Objective Function all hessian terms:")
        [[print(sp.simplify(obj_fn.diff(x).diff(y))) for x in [asym,A1sym,k11sym, k12sym, k21sym, k22sym,phisym]]
         for y in [asym,A1sym,k11sym, k12sym, k21sym, k22sym,phisym]]


#perform gradient descent on objective function, MSE((A1*cos(phase(k11,k12,k21,k22))-W)^2)
step = .01
max_its = 10000
i = 0
print("Init Vals:",a0, A10, k110, k120, k210, k220, phi0)
while np.linalg.norm(grad_obj(a0,A10,k110,k120,k210,k220, phi0))>1e-6 and i < max_its:
    curr = np.array([a0,A10,k110,k120,k210,k220,phi0])
    grad = grad_obj(curr[0],curr[1],curr[2],curr[3],curr[4],curr[5],curr[6])
    d = step
    new = curr - d*grad
    while obj(new[0],new[1],new[2],new[3],new[4],new[5],new[6])>obj(curr[0],curr[1],curr[2],curr[3],curr[4],curr[5],new[6]):
        print("Objective increased, decreasing step size")
        d*=.5
        new = curr - d * grad
        if d<1e-16:
            print("Norm of step size excessively small")
            print("Step: ", i)
            print("Gradient Norm", np.linalg.norm(grad))
            print("Obj Function Norm: ", np.linalg.norm(obj(a0,A10,k110, k120, k210, k220, phi0)))
            print("New Vals: ", a0, A10, k110, k120, k210, k220, phi0)
            break
    a0, A10, k110, k120, k210, k220, phi0 = new
    i += 1
    print("Step: ", i)
    print("Gradient Norm", np.linalg.norm(grad))
    print("Obj Function Norm: ",np.linalg.norm(obj(a0,A10,k110,k120,k210,k220,phi0)))
    print("New Vals: ", a0, A10, k110, k120, k210, k220, phi0)

#compare recovered pattern to given data
final_theta = a0*np.log(np.exp(k110*X+k120*Y) + np.exp(k210*X+k220*Y)) - phi0
final_pattern = A10*np.cos(final_theta)
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(W)
im1 = axs[1].imshow(final_pattern)
im2 = axs[2].imshow(np.abs(W-final_pattern))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Pattern, Approx Pattern, and Error")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/FieldEst_mu_{}.png".format(mu))
print("Est Field max err:", np.max(np.abs(W-final_pattern)))
print("Est Field mean err:", np.mean(np.abs(W-final_pattern)))




logfile.close()


