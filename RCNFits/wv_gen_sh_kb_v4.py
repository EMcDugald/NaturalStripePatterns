import numpy as np
from scipy.fft import fft2, fftfreq, fftshift, ifft2
import os
import sys
import time
from solve_sh import solve_sh_zigzag
import matplotlib.pyplot as plt

#########################################################################################################################
    ### Same as v2, but making a A1 a function of theta_x, theta_y
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

logfile = open(os.getcwd()+"/logs/sh_pgbs/wv_gen_sh_kb_v4_mu_{}.out".format(mu), 'w')
sys.stdout = logfile

start = time.time()

# option to print derivative terms using sympy
# option to set sign of amplitude
print_grad = True
print_hess = False
amp_pos = True

# set params for SH Solver
Nx = 256
tmax = 1000
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

def obj(k11,k12,phi):
    """
    function to be minimized
    """
    theta = np.log(np.exp(k11*X+k12*Y)+np.exp(k11*X-k12*Y))
    dtheta_dx = k11
    dtheta_dy = k12*np.tanh(Y*k12)
    k_sq = dtheta_dx**2 + dtheta_dy**2
    if amp_pos:
        amp1 = np.sqrt((4./3.)*(R-(k_sq-1)**2))
    else:
        amp1 = -np.sqrt((4. / 3.) * (R - (k_sq - 1) ** 2))
    return np.mean((amp1*np.cos(theta-phi)-W)**2)

def grad_obj(k11,k12,phi):
    """
    gradient of objective function
    """
    do_dk11 = np.mean(
        2.66666666666667 * (0.866025403784439 * W - np.sqrt(
            0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) * np.cos(
            phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))) * (
                    X * ((k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) * np.sin(
                phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + 2 * k11 * (
                                k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) * np.cos(
                phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))) / np.sqrt(
            0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2)

    )
    do_dk12 = np.mean(
        -2.66666666666667 * (0.866025403784439 * W - np.sqrt(
            0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) * np.cos(
            phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))) * (
                    Y * ((k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) * (
                        np.exp(X * k11 - Y * k12) - np.exp(X * k11 + Y * k12)) * np.sin(
                phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) - 2 * k12 * (
                                Y * k12 / np.cosh(Y * k12) ** 2 + np.tanh(Y * k12)) * (
                                np.exp(X * k11 - Y * k12) + np.exp(X * k11 + Y * k12)) * (
                                k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) * np.cos(
                phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) * np.tanh(Y * k12)) / (
                    np.sqrt(0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) * (
                        np.exp(X * k11 - Y * k12) + np.exp(X * k11 + Y * k12)))
    )
    do_dphi = np.mean(
        2.66666666666667 * np.sqrt(0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) * (
                    0.866025403784439 * W - np.sqrt(
                0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) * np.cos(
                phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))) * np.sin(
            phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))
    )
    return np.array([do_dk11,do_dk12,do_dphi])

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
phi0 = -np.pi/2


# optionally, use sympy to get derivatives of the objective function
if print_grad:
    import sympy as sp
    k11sym, k12sym, xsym, ysym, wsym, phisym = sp.symbols('k11,k12,X,Y,W,phi')
    thetasym = sp.log(sp.exp(k11sym*xsym + k12sym*ysym)+sp.exp(k11sym*xsym - k12sym*ysym))
    print("Phase function:", thetasym)
    [print("phase deriv in {}".format(str(x)), "\n", str(sp.simplify(thetasym.diff(x))).
           replace("exp", "np.exp").replace("log", "np.log")
           .replace("cos", "np.cos").replace("sin", "np.sin").replace("tan", "np.tan"))
     for x in [xsym, ysym]]
    dtheta_dxsym = k11sym
    dtheta_dysym = k12sym * sp.tanh(ysym * k12sym)
    k_sqsym = dtheta_dxsym**2 + dtheta_dysym**2
    if amp_pos:
        amp1sym = sp.sqrt((4./3.)*(R-(k_sqsym-1)**2))
    else:
        amp1sym = -sp.sqrt((4. / 3.) * (R - (k_sqsym - 1) ** 2))
    print("Amplitude Function:", amp1sym)
    obj_fnsym = (amp1sym * sp.cos(thetasym - phisym) - wsym) ** 2
    print("Objective Function:", obj_fnsym)
    print("Objective Function all gradient terms:")
    [print("deriv in {}".format(str(x)),"\n",str(sp.simplify(obj_fnsym.diff(x))).
           replace("exp","np.exp").replace("log","np.log").replace("sqrt","np.sqrt")
           .replace("cos","np.cos").replace("sin","np.sin").replace("tan","np.tan"))
            for x in [k11sym,k12sym,phisym]]
    if print_hess:
        print("Objective Function all hessian terms:")
        [[print(str(sp.simplify(obj_fnsym.diff(x).diff(y)))) for x in [k11sym, k12sym,phisym]]
         for y in [k11sym,k12sym,phisym]]


# #perform gradient descent on objective function, MSE((A1*cos(phase(k11,k12,k21,k22))-W)^2)
step = .01
max_its = 10000
i = 0
print("Init Vals:",k110, k120, phi0)
while np.linalg.norm(grad_obj(k110,k120,phi0))>1e-6 and i < max_its:
    curr = np.array([k110,k120,phi0])
    grad = grad_obj(curr[0],curr[1],curr[2])
    d = step
    new = curr - d*grad
    while obj(new[0],new[1],new[2])>obj(curr[0],curr[1],curr[2]):
        print("Objective increased, decreasing step size")
        d*=.5
        new = curr - d * grad
        if d<1e-16:
            print("Norm of step size excessively small")
            print("Step: ", i)
            print("Gradient Norm", np.linalg.norm(grad))
            print("Obj Function Norm: ", np.linalg.norm(obj(k110,k120,phi0)))
            print("New Vals: ", k110, k120, phi0)
            break
    k110, k120, phi0 = new
    i += 1
    print("Step: ", i)
    print("Gradient Norm", np.linalg.norm(grad))
    print("Obj Function Norm: ",np.linalg.norm(obj(k110,k120,phi0)))
    print("New Vals: ", k110, k120, phi0)

#compare recovered pattern to given data
final_theta = np.log(np.exp(k110*X+k120*Y) + np.exp(k110*X-k120*Y)) - phi0
final_dtheta_dx = k110
final_dtheta_dy = k120 * np.tanh(Y * k120)
final_k_sq = final_dtheta_dx**2 + final_dtheta_dy**2
if amp_pos:
    final_amp1 = np.sqrt((4. / 3.) * (R - (final_k_sq - 1) ** 2))
else:
    final_amp1 = -np.sqrt((4. / 3.) * (R - (final_k_sq - 1) ** 2))
final_pattern = final_amp1*np.cos(final_theta)
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(W)
im1 = axs[1].imshow(final_pattern)
im2 = axs[2].imshow(np.abs(W-final_pattern))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Pattern, Approx Pattern, and Error")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/FieldEst_v4_mu_{}.png".format(mu))
print("Est Field max err:", np.max(np.abs(W-final_pattern)))
print("Est Field mean err:", np.mean(np.abs(W-final_pattern)))

fig, ax = plt.subplots()
im0 = ax.imshow(final_theta)
plt.colorbar(im0,ax=ax)
plt.suptitle("Phase Estimate")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/PhaseEst_v4_mu_{}.png".format(mu))

fig, ax = plt.subplots()
im0 = ax.imshow(final_amp1)
plt.colorbar(im0,ax=ax)
plt.suptitle("Amp1 Estimate")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/Amp1Est_v4_mu_{}.png".format(mu))




logfile.close()


