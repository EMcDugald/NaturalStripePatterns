import numpy as np
from scipy.fft import fft2, fftfreq, fftshift, ifft2
import os
import sys
import time
from solve_sh import solve_sh_zigzag
import matplotlib.pyplot as plt

#########################################################################################################################
    ### Same as v2, but adding the amplitude A3 term to the ansatz
#########################################################################################################################


def get_centered_pattern(surf,x,y):
    """
    SH solver for knee bends uses an awkward domain. This method retrieves a symmetric PGB and normalizes the coordinates
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

logfile = open(os.getcwd()+"/logs/sh_pgbs/wv_gen_sh_kb_v3_mu_{}.out".format(mu), 'w')
sys.stdout = logfile

start = time.time()

# option to print derivative terms using sympy
print_grad = True
print_hess = False

# set params for SH Solver
Nx = 256
tmax = 200
R = .5
h = 1
a = 1.0

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

def obj(A1,A3,k11,k12,phi):
    """
    function to be minimized
    """
    theta = a*np.log(np.exp(k11*X+k12*Y)+np.exp(k11*X-k12*Y))
    return np.mean((A1*np.cos(theta-phi) + A3*np.cos(3*(theta-phi))-W)**2)

def grad_obj(A1,A3,k11,k12,phi):
    """
    gradient of objective function
    """
    do_dA1 = np.mean(
        2 * (A1 * np.cos(phi - 1.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + A3 * np.cos(
            3 * phi - 3.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) - W) * np.cos(
            phi - 1.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))
    )
    do_dA3 = np.mean(
        2 * (A1 * np.cos(phi - 1.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + A3 * np.cos(
            3 * phi - 3.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) - W) * np.cos(
            3 * phi - 3.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))
    )
    do_dk11 = np.mean(
        X * (2.0 * A1 * np.sin(phi - 1.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + 6.0 * A3 * np.sin(
            3 * phi - 3.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))) * (
                    A1 * np.cos(phi - 1.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + A3 * np.cos(
                3 * phi - 3.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) - W)
    )
    do_dk12 = np.mean(
        -Y*(2.0*A1*np.sin(phi - 1.0*np.log(2*np.exp(X*k11)*np.cosh(Y*k12))) +
            6.0*A3*np.sin(3*phi - 3.0*np.log(2*np.exp(X*k11)*np.cosh(Y*k12))))*
        (np.exp(X*k11 - Y*k12) - np.exp(X*k11 + Y*k12))*
        (A1*np.cos(phi - 1.0*np.log(2*np.exp(X*k11)*np.cosh(Y*k12)))
         + A3*np.cos(3*phi - 3.0*np.log(2*np.exp(X*k11)*np.cosh(Y*k12))) - W)
        /(np.exp(X*k11 - Y*k12) + np.exp(X*k11 + Y*k12))
    )
    do_dphi = np.mean(
        -2 * (A1 * np.sin(phi - 1.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + 3 * A3 * np.sin(
            3 * phi - 3.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))) * (
                    A1 * np.cos(phi - 1.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + A3 * np.cos(
                3 * phi - 3.0 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) - W)
    )
    return np.array([do_dA1,do_dA3,do_dk11,do_dk12,do_dphi])

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
A10 = np.max(np.abs(W))
print("Max abs W:",np.max(np.abs(W)))
phi0 = np.pi/4
A30 = A10**3/((4./3.)*(R-64.))


# optionally, use sympy to get derivatives of the objective function
if print_grad:
    import sympy as sp
    A1sym, A3sym, k11sym, k12sym, xsym, ysym, wsym, phisym = sp.symbols('A1,A3,k11,k12,X,Y,W, phi')
    obj_fn = (A1sym*sp.cos(a*sp.log(sp.exp(k11sym*xsym + k12sym*ysym)+sp.exp(k11sym*xsym - k12sym*ysym)) - phisym)
              + A3sym*sp.cos(3*(a*sp.log(sp.exp(k11sym*xsym + k12sym*ysym)+sp.exp(k11sym*xsym - k12sym*ysym)) - phisym))
              - wsym)**2
    print("Objective Function all gradient terms:")
    [print("deriv in {}".format(str(x)),"\n",str(sp.simplify(obj_fn.diff(x))).
           replace("exp","np.exp").replace("log","np.log")
           .replace("cos","np.cos").replace("sin","np.sin"))
            for x in [A1sym,A3sym,k11sym,k12sym,phisym]]
    if print_hess:
        print("Objective Function all hessian terms:")
        [[print(str(sp.simplify(obj_fn.diff(x).diff(y)))) for x in [A1sym,A3sym,k11sym, k12sym,phisym]]
         for y in [A1sym,A3sym,k11sym,k12sym,phisym]]


#perform gradient descent on objective function, MSE((A1*cos(phase(k11,k12,k21,k22))-W)^2)
step = .01
max_its = 10000
i = 0
print("Init Vals:",A10, A30, k110, k120, phi0)
while np.linalg.norm(grad_obj(A10,A30,k110,k120,phi0))>1e-6 and i < max_its:
    curr = np.array([A10,A30,k110,k120,phi0])
    grad = grad_obj(curr[0],curr[1],curr[2],curr[3],curr[4])
    d = step
    new = curr - d*grad
    while obj(new[0],new[1],new[2],new[3],new[4])>obj(curr[0],curr[1],curr[2],curr[3],curr[4]):
        print("Objective increased, decreasing step size")
        d*=.5
        new = curr - d * grad
        if d<1e-16:
            print("Norm of step size excessively small")
            print("Step: ", i)
            print("Gradient Norm", np.linalg.norm(grad))
            print("Obj Function Norm: ", np.linalg.norm(obj(A10,A30,k110,k120,phi0)))
            print("New Vals: ", A10, A30, k110, k120, phi0)
            break
    A10, A30, k110, k120, phi0 = new
    i += 1
    print("Step: ", i)
    print("Gradient Norm", np.linalg.norm(grad))
    print("Obj Function Norm: ",np.linalg.norm(obj(A10,A30,k110,k120,phi0)))
    print("New Vals: ", A10, A30, k110, k120, phi0)

#compare recovered pattern to given data
final_theta = a*np.log(np.exp(k110*X+k120*Y) + np.exp(k110*X-k120*Y)) - phi0
final_pattern = A10*np.cos(final_theta) + A30*np.cos(3*final_theta)
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(W)
im1 = axs[1].imshow(final_pattern)
im2 = axs[2].imshow(np.abs(W-final_pattern))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Pattern, Approx Pattern, and Error")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/FieldEst_v3_mu_{}.png".format(mu))
print("Est Field max err:", np.max(np.abs(W-final_pattern)))
print("Est Field mean err:", np.mean(np.abs(W-final_pattern)))

fig, ax = plt.subplots()
im0 = ax.imshow(final_theta)
plt.colorbar(im0,ax=ax)
plt.suptitle("Phase Estimate")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/PhaseEst_v3_mu_{}.png".format(mu))




logfile.close()


