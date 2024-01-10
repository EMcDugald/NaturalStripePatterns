import numpy as np
from scipy.fft import fft2, fftfreq, fftshift, ifft2
import os
import sys
import time
from solve_sh import solve_sh_zigzag
import matplotlib.pyplot as plt
from utils import t6hat
import math
import scipy.io as sio


#########################################################################################################################
    ### Same as v4, but also includes A3
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

def sigma(rmax_x,rmin_x,rmax_y,rmin_y,xshift,yshift):
    """
    makes a smooth indicator function
    """
    return t6hat(rmax_x, rmin_x, X - xshift) * t6hat(rmax_y, rmin_y, Y - yshift)


# mu determines sharpness of knee bend
mu = .3

logfile = open(os.getcwd()+"/logs/sh_pgbs/wv_gen_sh_kb_v5_mu_{}.out".format(mu), 'w')
sys.stdout = logfile

start = time.time()

# option to print derivative terms using sympy
# option to set sign of amplitude
print_grad = False
print_hess = False
amp_pos = True

# set params for SH Solver
Nx = 512
tmax = 1000
R = .5
h = 1

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
    amp3 = (amp1**3)/(4*(R-(9*k_sq-1)**2))
    return np.mean((amp1*np.cos(theta-phi) + amp3*np.cos(3*(theta-phi))-W)**2)

def grad_obj(k11,k12,phi):
    """
    gradient of objective function
    """
    do_dk11 = np.mean(
        0.148148148148148 * ((0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) ** (3 / 2) * np.cos(
            3 * phi - 3 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + (0.649519052838329 * W - 0.75 * np.sqrt(
            0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) * np.cos(
            phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))) * (
                                         4 * (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 2.0)) * (
                    6 * X * ((k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) ** 2 * (
                        4 * (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 2.0) * np.sin(
                3 * phi - 3 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + 24.0 * X * (
                                (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) * (
                                (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) ** 2 * np.sin(
                phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) - 288 * k11 * (
                                (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) ** 2 * (
                                9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) * np.cos(
                3 * phi - 3 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + 12 * k11 * (
                                (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) * (
                                4 * (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 2.0) * (
                                k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) * np.cos(
                3 * phi - 3 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + 48.0 * k11 * (
                                (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) ** 2 * (
                                k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) * np.cos(
                phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))) / (
                    np.sqrt(0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) * (
                        (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) ** 2 * (
                                4 * (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 2.0))
    )
    do_dk12 = np.mean(
        -0.0740740740740741 * ((0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) ** (3 / 2) * np.cos(
            3 * phi - 3 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + (0.649519052838329 * W - 0.75 * np.sqrt(
            0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) * np.cos(
            phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))) * (4 * (
                    9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 2.0)) * (
                    12 * Y * ((k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) ** 2 * (
                        4 * (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 2.0) * (
                                np.exp(X * k11 - Y * k12) - np.exp(X * k11 + Y * k12)) * np.sin(
                3 * phi - 3 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + 48.0 * Y * (
                                (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) * (
                                (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) ** 2 * (
                                np.exp(X * k11 - Y * k12) - np.exp(X * k11 + Y * k12)) * np.sin(
                phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + 576 * k12 * (
                                Y * k12 / np.cosh(Y * k12) ** 2 + np.tanh(Y * k12)) * (
                                (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) ** 2 * (
                                np.exp(X * k11 - Y * k12) + np.exp(X * k11 + Y * k12)) * (
                                9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) * np.cos(
                3 * phi - 3 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) * np.tanh(Y * k12) - 24 * k12 * (
                                Y * k12 / np.cosh(Y * k12) ** 2 + np.tanh(Y * k12)) * (
                                (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) * (
                                4 * (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 2.0) * (
                                np.exp(X * k11 - Y * k12) + np.exp(X * k11 + Y * k12)) * (
                                k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) * np.cos(
                3 * phi - 3 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) * np.tanh(Y * k12) - 96.0 * k12 * (
                                Y * k12 / np.cosh(Y * k12) ** 2 + np.tanh(Y * k12)) * (
                                (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) ** 2 * (
                                np.exp(X * k11 - Y * k12) + np.exp(X * k11 + Y * k12)) * (
                                k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) * np.cos(
                phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) * np.tanh(Y * k12)) / (
                    np.sqrt(0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) * (
                        (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) ** 2 * (
                                4 * (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 2.0) * (
                                np.exp(X * k11 - Y * k12) + np.exp(X * k11 + Y * k12)))

    )
    do_dphi = np.mean(
        0.148148148148148 * np.sqrt(0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) * (
                    (0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) ** (3 / 2) * np.cos(
                3 * phi - 3 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + (0.649519052838329 * W - 0.75 * np.sqrt(
                0.5 - (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2) * np.cos(
                phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))) * (
                                4 * (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 2.0)) * (
                    (6 * (k11 ** 2 + k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 3.0) * np.sin(
                3 * phi - 3 * np.log(2 * np.exp(X * k11) * np.cosh(Y * k12))) + (
                                6.0 * (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 3.0) * np.sin(
                phi - np.log(2 * np.exp(X * k11) * np.cosh(Y * k12)))) / (
                    (9 * k11 ** 2 + 9 * k12 ** 2 * np.tanh(Y * k12) ** 2 - 1) ** 2 - 0.5) ** 2

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
phi0 = -np.pi/3


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
    amp3sym = (amp1sym ** 3) / (4 * (R - (9 * k_sqsym - 1) ** 2))
    print("Amplitude 1 Function:", amp1sym)
    print("Amplitude 3 Function:", amp1sym)
    obj_fnsym = (amp1sym * sp.cos(thetasym - phisym) + amp3sym*sp.cos(3*(thetasym-phisym)) - wsym) ** 2
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


#perform gradient descent on objective function, MSE((A1*cos(phase(k11,k12,k21,k22))-W)^2)
step = .01
max_its = 5000
i = 0
print("Init Vals:",k110, k120, phi0)
while np.linalg.norm(grad_obj(k110,k120,phi0))>1e-4 and i < max_its:
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
final_amp3 = (final_amp1**3)/(4*(R-(9*final_k_sq-1)**2))
final_pattern = final_amp1*np.cos(final_theta) + final_amp3*np.cos(3*final_theta)
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(W)
im1 = axs[1].imshow(final_pattern)
im2 = axs[2].imshow(np.abs(W-final_pattern))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Pattern, Approx Pattern, and Error")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/FieldEst_v5_mu_{}.png".format(mu))
print("Est Field max err:", np.max(np.abs(W-final_pattern)))
print("Est Field mean err:", np.mean(np.abs(W-final_pattern)))

fig, ax = plt.subplots()
im0 = ax.imshow(final_theta)
plt.colorbar(im0,ax=ax)
plt.suptitle("Phase Estimate")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/PhaseEst_v5_mu_{}.png".format(mu))

fig, axs = plt.subplots(nrows=1,ncols=2)
im0 = axs[0].imshow(final_amp1)
im1 = axs[1].imshow(final_amp3)
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Amp1 and Amp3 Estimate")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/AmpEsts_v5_mu_{}.png".format(mu))


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
fig, axs = plt.subplots(nrows=1,ncols=2)
im0 = axs[0].imshow(theta_x_approx)
im1 = axs[1].imshow(theta_y_approx)
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Found Phase Gradient")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/PhaseGradients_v5_mu_{}.png".format(mu))

#compare recovered wave nums to exact wave nums
wavenums_approx = np.sqrt(theta_x_approx**2+theta_y_approx**2)
fig, axs = plt.subplots()
im0 = axs.imshow(wavenums_approx)
plt.colorbar(im0,ax=axs)
plt.suptitle("Found Wave Nums")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/WaveNums_v5_mu_{}.png".format(mu))

#compare recovered divk to exact divk
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
im0 = axs[0].imshow(divk_approx)
im1 = axs[1].imshow(curlk_approx)
im2 = axs[2].imshow(Jk_approx)
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Div(k), Curl(k), J(k)")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_pgbs/DivKCurlkJk_v5_mu_{}.png".format(mu))


mdict = {'theta_x_approx':theta_x_approx,'theta_y_approx': theta_y_approx,
         'theta_xx_approx': theta_xx_approx, 'theta_yy_approx': theta_yy_approx,
         'theta_xy_approx':theta_xy_approx,'theta_yx_approx':theta_yx_approx,
         'divk_approx': divk_approx,'curlk_approx': curlk_approx,'Jk_approx':Jk_approx,
         'wavenums_approx':wavenums_approx,'Xinterior':Xinterior,'Yinterior':Yinterior}

sio.savemat(os.getcwd()+"/data/sh_pgbs/v5_mu_{}.mat".format(mu),mdict)
logfile.close()


logfile.close()


