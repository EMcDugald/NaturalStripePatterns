import numpy as np
from scipy.fft import fft2, fftfreq, fftshift, ifft2
import os
import matplotlib.pyplot as plt
import sys
import time
from utils import t6hat
import math
import scipy.io as sio

#########################################################################################################################
    ### This script compute wave vector fields from knee bends using the RCN knee bend ansatz ###
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


# mu determines sharpness of knee bend
#mu = .3
#mu = .5
#mu = .7
mu = .9


logfile = open(os.getcwd()+"/logs/knee_bends/wv_gen_kb_mu_{}.out".format(mu), 'w')
sys.stdout = logfile

start = time.time()

# option to print derivative terms using sympy
print_grad = True
print_hess = False

# set up geometry and parameters for pattern
Nx = 512
Ny = 512
kx = np.sqrt(1-mu**2)
ky = mu
Ly = 16*np.pi/kx
Lx = Ly
xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
X,Y = np.meshgrid(xx,yy)
a = 1.0
ss_factor = 4
len_scale = .75

print("Grid Dims:", "Nx = ",Nx, "Ny = ",Ny)
print("Dom Size:", "Lx = ",Lx, "Ly = ", Ly)
print("Approximation length scale:", len_scale)
print("Approximation subsampling:", ss_factor)


# define functions for exact phase, exact phase derivatives, gaussian bump, objective function, and objective function gradient
def theta(k1,k2):
    """
    phase
    """
    return k1*X + np.log(2*np.cosh(k2*Y))

def theta_x(x):
    """
    partial derivative in x of phase
    """
    return kx*np.ones(shape=np.shape(x))

def theta_y(y):
    """
    partial derivative in y of phase
    """
    return ky*np.tanh(ky*y)

def divk(y):
    """
    divergence of wave vector (aka, laplacian of phase)
    """
    return ky**2/(np.cosh(ky*y)**2)

def curlk():
    """
    curl of wave vector
    """
    return np.zeros(shape=np.shape(X))

def Jk():
    """
    jacobian determinant of wave vector (aka, hessian determinant of phase)
    """
    return np.zeros(shape=np.shape(X))

def gaussian(x0,y0,X,Y,sigma):
    """
    gaussian bump
    """
    exponent = (X-x0)**2 + (Y-y0)**2
    return np.exp(-exponent/(sigma**2))

def obj(k11,k12,k21,k22):
    """
    function to be minimized
    """
    theta = a*np.log(np.exp(k11*X+k12*Y)+np.exp(k21*X+k22*Y))
    return np.mean((np.cos(theta)-W)**2)

def grad_obj(k11,k12,k21,k22):
    """
    gradient of objective function
    """
    do_dk11 = np.mean(
        2*a*X*(W - np.cos(a*np.log(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y))))
        *np.exp(k11*X + k12*Y)*np.sin(a*np.log(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y)))
        /(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y))
    )
    do_dk12 = np.mean(
        2*a*Y*(W - np.cos(a*np.log(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y))))
        *np.exp(k11*X + k12*Y)*np.sin(a*np.log(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y)))
        /(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y)))
    do_dk21 = np.mean(
        2*a*X*(W - np.cos(a*np.log(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y))))
        *np.exp(k21*X + k22*Y)*np.sin(a*np.log(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y)))
        /(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y))
    )
    do_dk22 = np.mean(
        2*a*Y*(W - np.cos(a*np.log(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y))))
        *np.exp(k21*X + k22*Y)*np.sin(a*np.log(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y)))
        /(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y))
    )
    return np.array([do_dk11, do_dk12, do_dk21, do_dk22])

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




# compute exact phase, exact pattern, and exact phase gradient
theta = theta(kx,ky)
W = np.cos(theta)
theta_x_exact = theta_x(X)
theta_y_exact = theta_y(Y)


# make frequency grid
xi, eta = freq_grids(Lx,Nx,Ly,Ny)


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


# optionally, use sympy to get derivatives of the objective function
if print_grad:
    import sympy as sp
    k11sym, k12sym, k21sym, k22sym, xsym, ysym, wsym = sp.symbols('k11,k12,k21,k22,x,y,w')
    obj_fn = (sp.cos(a*sp.log(sp.exp(k11sym*xsym + k12sym*ysym)+sp.exp(k21sym*xsym+k22sym*ysym)))-wsym)**2
    phase_fn = k11sym*xsym + sp.log(2*sp.cosh(k12sym*ysym))
    print("Objective Function:", sp.simplify(obj_fn))
    print("Objective Function Analytical Derivative in k11:", sp.simplify(obj_fn.diff(k11sym)))
    print("Objecctive Function Analytical Derivative in k12:", sp.simplify(obj_fn.diff(k12sym)))
    print("Objective Function Analytical Derivative in k21:", sp.simplify(obj_fn.diff(k21sym)))
    print("Objective Function Analytical Derivative in k22:", sp.simplify(obj_fn.diff(k22sym)))
    print("Phase Function Partial in x:", sp.simplify(phase_fn.diff(xsym)))
    print("Phase Function Partial in y:", sp.simplify(phase_fn.diff(ysym)))
    print("Phase Function Partial in xx:", sp.simplify(phase_fn.diff(xsym).diff(xsym)))
    print("Phase Function Partial in yy:", sp.simplify(phase_fn.diff(ysym).diff(ysym)))
    print("Phase Function Partial in xy:", sp.simplify(phase_fn.diff(xsym).diff(ysym)))
    print("Phase Function Partial in yx:", sp.simplify(phase_fn.diff(ysym).diff(xsym)))
    print("Phase Function all gradient terms:")
    [print(sp.simplify(obj_fn.diff(x))) for x in [k11sym,k12sym,k21sym,k22sym]]
    if print_hess:
        print("Objective Function all hessian terms:")
        [[print(sp.simplify(obj_fn.diff(x).diff(y))) for x in [k11sym, k12sym, k21sym, k22sym]]
         for y in [k11sym, k12sym, k21sym, k22sym]]


# perform gradient descent on objective function, MSE(cos(phase(k11,k12,k21,k22))-W)^2)
step = .05
max_its = 5000
i = 0
print("Init Vals:",k110, k120, k210, k220)
while np.linalg.norm(grad_obj(k110,k120,k210,k220))>1e-10 and i < max_its:
    curr = np.array([k110,k120,k210,k220])
    grad = grad_obj(curr[0],curr[1],curr[2],curr[3])
    d = step
    new = curr - d*grad
    while obj(new[0],new[1],new[2],new[3])>obj(curr[0],curr[1],curr[2],curr[3]):
        print("Objective increased, decreasing step size")
        d*=.5
        new = curr - d * grad
        if d<1e-16:
            print("Norm of step size excessively small")
            print("Step: ", i)
            print("Gradient Norm", np.linalg.norm(grad))
            print("Obj Function Norm: ", np.linalg.norm(obj(k110, k120, k210, k220)))
            print("New Vals: ", k110, k120, k210, k220)
            break
    k110, k120, k210, k220 = new
    i += 1
    print("Step: ", i)
    print("Gradient Norm", np.linalg.norm(grad))
    print("Obj Function Norm: ",np.linalg.norm(obj(k110,k120,k210,k220)))
    print("New Vals: ", k110, k120, k210, k220)

print("exact kx:",kx)
print("exact ky:",ky)
print("found kx:",k110)
print("found ky:",k120)

#compare recovered pattern to given data
final_theta = a*np.log(np.exp(k110*X+k120*Y) + np.exp(k210*X+k220*Y))
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
plt.savefig(os.getcwd()+"/figs/knee_bends/FieldEst_mu_{}.png".format(mu))
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
        # G = gaussian(X[r, c], Y[r, c], X, Y, 1.25)
        # f = G*final_theta
        # s = sigma(7,2,7,2,X[r,c],Y[r,c])
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
divk_exact = divk(Y)
divk_exact_ss = divk_exact[rows[0]:rows[-1]+ss_factor,cols[0]:cols[-1]+ss_factor][::ss_factor,::ss_factor]
curlk_exact = curlk()
curlk_exact_ss = curlk_exact[rows[0]:rows[-1]+ss_factor,cols[0]:cols[-1]+ss_factor][::ss_factor,::ss_factor]
Jk_exact = Jk()
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
plt.savefig(os.getcwd()+"/figs/knee_bends/PhaseGradients_mu_{}.png".format(mu))
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
plt.savefig(os.getcwd()+"/figs/knee_bends/WaveNums_mu_{}.png".format(mu))
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
plt.savefig(os.getcwd()+"/figs/knee_bends/DivK_mu_{}.png".format(mu))
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
plt.savefig(os.getcwd()+"/figs/knee_bends/CurlK_mu_{}.png".format(mu))
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
plt.savefig(os.getcwd()+"/figs/knee_bends/Jk_mu_{}.png".format(mu))
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

sio.savemat(os.getcwd()+"/data/knee_bends/mu_{}.mat".format(mu),mdict)
logfile.close()

# #ToDo: 1) mlpement SINDy fit,
# #ToDo: 2) Repeat for concave disclination (sum of 6 exponentials)
# #ToDo: 3) Repeat for stationary dislocation
# # Note: make separate methods for grad descent, local spectrum derivative calcs, plotting


