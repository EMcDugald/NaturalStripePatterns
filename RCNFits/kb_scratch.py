import numpy as np
from scipy.fft import fft2, fftfreq, fftshift, ifft2
import os
import matplotlib.pyplot as plt
import sys
import time
from derivatives import FiniteDiffDerivs



#mu = .3
#mu = .5
#mu = .7
mu = .9

logfile = open(os.getcwd()+"/logs/scratch/kb_scratch_{}.out".format(mu), 'w')
sys.stdout = logfile

print_derivs = False

### DEFINTE GEOMETRY, PHASE, AND PATTERN (WHICH IS COS(PHASE))
Nx = 256
Ny = 256
kx = np.sqrt(1-mu**2)
ky = mu

Ly = 10*np.pi/kx
Lx = Ly
xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
X,Y = np.meshgrid(xx,yy)
theta = kx*X + np.log(2*np.cosh(ky*Y))
W = np.cos(theta)
k1 = np.array([kx,ky])
k2 = np.array([kx,-ky])

### PLOT THE PATTERN
fig, ax = plt.subplots()
im0 = ax.imshow(W)
plt.colorbar(im0,ax=ax)
plt.suptitle("Pattern")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/scratch/Pattern_mu_{}.png".format(mu))

### MAKE SURE THE ANSATZ FOR OPTIMIZATION MATCHES THE ANALYTICAL SD SOLUTION
fig, axs = plt.subplots(nrows=3,ncols=1,figsize=(6,20))
im0 = axs[0].imshow(theta)
im1 = axs[1].imshow(np.log(np.exp(k1[0]*X+k1[1]*Y) + np.exp(k2[0]*X+k2[1]*Y)))
im2 = axs[2].imshow(np.abs(theta-np.log(np.exp(k1[0]*X+k1[1]*Y) + np.exp(k2[0]*X+k2[1]*Y))))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Phase, Phase Ansatz, and Error")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/scratch/PhaseAnsatz_mu_{}.png".format(mu))
print("theta ansatz max err:", np.max(np.abs(theta-np.log(np.exp(k1[0]*X+k1[1]*Y) + np.exp(k2[0]*X+k2[1]*Y)))))


### COMPUTE EXACT PHASE GRADIENT COMPONENTS
def theta_x(x,y):
    return kx*np.ones(shape=np.shape(x))

def theta_y(x,y):
    return ky*np.tanh(ky*y)

dx = xx[1]-xx[0]
dy = yy[1]-yy[0]
print("grid spacing:",dx,dy)

### COMPUTE FINITE DIFFERENCE PHASE GRADIENT COMPONENTS
theta_x_exact = theta_x(X,Y)
theta_y_exact = theta_y(X,Y)
theta_x_approx = FiniteDiffDerivs(theta,dx,dy,type='x')
theta_y_approx = FiniteDiffDerivs(theta,dx,dy,type='y')

# ### COMPARE ERROR OF EXACT VS FINITE DIFF PHASE GRADIENTS
fig, axs = plt.subplots(nrows=2,ncols=2)
im0 = axs[0,0].imshow(theta_x_exact[1:-1,1:-1])
im1 = axs[0,1].imshow(theta_y_exact[1:-1,1:-1])
im2 = axs[1,0].imshow(theta_x_approx)
im3 = axs[1,1].imshow(theta_y_approx)
plt.colorbar(im0,ax=axs[0,0])
plt.colorbar(im1,ax=axs[0,1])
im2.set_clim(np.min(theta_x_exact[1:-1,1:-1]),np.max(theta_x_exact[1:-1,1:-1]))
im3.set_clim(np.min(theta_y_exact[1:-1,1:-1]),np.max(theta_y_exact[1:-1,1:-1]))
plt.colorbar(im2,ax=axs[1,0])
plt.colorbar(im3,ax=axs[1,1])
plt.suptitle("Exact vs FD Phase Grad")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/scratch/InitPhaseGradients_mu_{}.png".format(mu))
print("FD theta_x max err:", np.max(np.abs(theta_x_exact[1:-1,1:-1]-theta_x_approx)))
print("FD theta_y max err:", np.max(np.abs(theta_y_exact[1:-1,1:-1]-theta_y_approx)))

### ESTIMATE UPPER PLANE WAVE VEC WITH LOCAL SPECTRUM

kxx = (2. * np.pi / Lx) * fftfreq(Nx, 1. / Nx)  # wave numbers
kyy = (2. * np.pi / Ly) * fftfreq(Ny, 1. / Ny)
xi, eta = np.meshgrid(kxx, kyy)

def Gaussian(x0,y0,X,Y,sigma):
    exponent = (X-x0)**2 + (Y-y0)**2
    return np.exp(-exponent/(sigma**2))

a0 = 1.0
G = Gaussian(X[0, int(Nx/2)], Y[int(Ny/4), 0], X, Y, 3.3)
f = G*W
spec = fftshift(fft2(f))
max_spec_idx = np.argsort(-np.abs(spec).flatten())[0]
kx0 = np.abs(fftshift(xi).flatten()[max_spec_idx])
ky0 = np.abs(fftshift(eta).flatten()[max_spec_idx])
k110 = kx0
k120 = ky0
k210 = kx0
k220 = -ky0
theta0 = a0*np.log(np.exp(k110*X+k120*Y)+np.exp(k210*X+k220*Y))
fig, axs = plt.subplots(nrows=3,ncols=1, figsize=(6,20))
im0 = axs[0].imshow(G)
plt.colorbar(im0,ax=axs[0])
im1 = axs[1].imshow(f*G)
plt.colorbar(im1,ax=axs[1])
im2 = axs[2].imshow(np.cos(theta0))
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Pattern Initial Guess")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/scratch/PatternInitialGuess_mu_{}.png".format(mu))


### OPTIMIZATION WHERE a IS SET TO 1
a = 1.0

def obj(k11,k12,k21,k22):
    theta = a*np.log(np.exp(k11*X+k12*Y)+np.exp(k21*X+k22*Y))
    return np.mean((np.cos(theta)-W)**2)

if print_derivs:
    import sympy as sp
    k11sym, k12sym, k21sym, k22sym, xsym, ysym, wsym = sp.symbols('k11,k12,k21,k22,x,y,w')
    expr = (sp.cos(a*sp.log(sp.exp(k11sym*xsym + k12sym*ysym)+sp.exp(k21sym*xsym+k22sym*ysym)))-wsym)**2
    print("Objective Function:", sp.simplify(expr))
    print("Analytical Derivative in k11:", sp.simplify(expr.diff(k11sym)))
    print("Analytical Derivative in k12:", sp.simplify(expr.diff(k12sym)))
    print("Analytical Derivative in k21:", sp.simplify(expr.diff(k21sym)))
    print("Analytical Derivative in k22:", sp.simplify(expr.diff(k22sym)))
    print("All gradient terms:")
    [print(sp.simplify(expr.diff(x))) for x in [k11sym,k12sym,k21sym,k22sym]]
    print("All hessian terms:")
    [[print(sp.simplify(expr.diff(x).diff(y))) for x in [k11sym,k12sym,k21sym,k22sym]]
                                    for y in [k11sym,k12sym,k21sym,k22sym]]

def grad_obj(k11,k12,k21,k22):
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

step = .05
max_its = 5000
i = 0
print("Init Vals:",k110, k120, k210, k220)
while np.linalg.norm(grad_obj(k110,k120,k210,k220))>1e-6 and i < max_its:
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

final_theta = a*np.log(np.exp(k1[0]*X+k1[1]*Y) + np.exp(k2[0]*X+k2[1]*Y))
final_pattern = np.cos(final_theta)
fig, axs = plt.subplots(nrows=3,ncols=1,figsize=(6,20))
im0 = axs[0].imshow(W)
im1 = axs[1].imshow(final_pattern)
im2 = axs[2].imshow(np.abs(W-final_pattern))
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("Pattern, Approx Pattern, and Error")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/scratch/FieldEst_mu_{}.png".format(mu))
print("Est Field max err:", np.max(np.abs(W-final_pattern)))

### COMPUTING K Fields from Phase ###
### Exact Derivatives ###
if print_derivs:
    k11sym, k12sym, k21sym, k22sym, xsym, ysym = sp.symbols('k11,k12,k21,k22,x,y')
    expr2 = a*sp.log(sp.exp(k11sym*xsym + k12sym*ysym)+sp.exp(k21sym*xsym+k22sym*ysym))
    print("Analytical Phase Derivative in x:", sp.simplify(expr2.diff(xsym)))
    print("Analytical Phase Derivative in y:", sp.simplify(expr2.diff(ysym)))

def dphase_dx(k11,k12,k21,k22):
    return a * (k11 * np.exp(k11 * X + k12 * Y) + k21 * np.exp(k21 * X + k22 * Y)) \
           / (np.exp(k11 * X + k12 * Y) + np.exp(k21 * X + k22 * Y))

def dphase_dy(k11,k12,k21,k22):
    return a*(k12*np.exp(k11*X + k12*Y) + k22*np.exp(k21*X + k22*Y))\
           /(np.exp(k11*X + k12*Y) + np.exp(k21*X + k22*Y))

theta_x_exact_ansatz = dphase_dx(k110,k120,k210,k220)
theta_y_exact_ansatz = dphase_dy(k110,k120,k210,k220)

fig, axs = plt.subplots(nrows=2,ncols=2)
im0 = axs[0,0].imshow(theta_x_exact)
im1 = axs[0,1].imshow(theta_y_exact)
im2 = axs[1,0].imshow(theta_x_exact_ansatz)
im3 = axs[1,1].imshow(theta_y_exact_ansatz)
plt.colorbar(im0,ax=axs[0,0])
plt.colorbar(im1,ax=axs[0,1])
im2.set_clim(np.min(theta_x_exact),np.max(theta_x_exact))
plt.colorbar(im2,ax=axs[1,0])
im3.set_clim(np.min(theta_y_exact),np.max(theta_y_exact))
plt.colorbar(im3,ax=axs[1,1])
plt.suptitle("Exact vs Ansatz Phase Grad")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/scratch/PhaseGradAnsatz_mu_{}.png".format(mu))
print("Ansatz theta_x max err:", np.max(np.abs(theta_x_exact-theta_x_exact_ansatz)))
print("Ansatz theta_y max err:", np.max(np.abs(theta_y_exact-theta_y_exact_ansatz)))


theta_x_finite_diff = FiniteDiffDerivs(final_theta,dx,dy,type='x')
theta_y_finite_diff = FiniteDiffDerivs(final_theta,dx,dy,type='y')

fig, axs = plt.subplots(nrows=2,ncols=2)
im0 = axs[0,0].imshow(theta_x_exact[1:-1,1:-1])
im1 = axs[0,1].imshow(theta_y_exact[1:-1,1:-1])
im2 = axs[1,0].imshow(theta_x_finite_diff)
im3 = axs[1,1].imshow(theta_y_finite_diff)
plt.colorbar(im0,ax=axs[0,0])
im2.set_clim(np.min(theta_x_exact[1:-1,1:-1]),np.max(theta_x_exact[1:-1,1:-1]))
plt.colorbar(im1,ax=axs[0,1])
plt.colorbar(im2,ax=axs[1,0])
im3.set_clim(np.min(theta_y_exact[1:-1,1:-1]),np.max(theta_y_exact[1:-1,1:-1]))
plt.colorbar(im3,ax=axs[1,1])
plt.suptitle("Exact vs FD Phase Grad")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/scratch/PhaseGradFD_mu_{}.png".format(mu))

print("FiniteDiff theta_x max err:", np.max(np.abs(theta_x_exact[1:-1,1:-1]-theta_x_finite_diff)))
print("FiniteDiff theta_y max err:", np.max(np.abs(theta_y_exact[1:-1,1:-1]-theta_y_finite_diff)))


col_indices = np.where((X[0,:]>-Lx/4) & (X[0,:]<Lx/4))[0]
row_indices = np.where((Y[:,0]>-Ly/4) & (Y[:,0]<Ly/4))[0]
innerThetaAnsatz = final_theta[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1]

theta_x_loc_spec = np.zeros(np.shape(innerThetaAnsatz))
theta_y_loc_spec = np.zeros(np.shape(innerThetaAnsatz))
print("Shape of sampled local spec derivs:",np.shape(theta_y_loc_spec))

r_shift = row_indices[0]
c_shift = col_indices[0]
for r in row_indices:
    for c in col_indices:
        G = Gaussian(X[r, c], Y[r, c], X, Y, 1.25)
        f = G*final_theta
        dfdx = np.real(ifft2(1j*xi*fft2(f)))
        theta_x_loc_spec[int(r - r_shift), int(c - c_shift)] += dfdx[r,c]
        dfdy = np.real(ifft2(1j*eta*fft2(f)))
        theta_y_loc_spec[int(r - r_shift), int(c - c_shift)] += dfdy[r, c]


fig, axs = plt.subplots(nrows=2,ncols=2)
im0 = axs[0,0].imshow(theta_x_exact[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1])
im1 = axs[0,1].imshow(theta_y_exact[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1])
im2 = axs[1,0].imshow(theta_x_loc_spec)
im3 = axs[1,1].imshow(theta_y_loc_spec)
plt.colorbar(im0,ax=axs[0,0])
im2.set_clim(
    np.min(theta_x_exact[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1]),
             np.max(theta_x_exact[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1])
)
plt.colorbar(im1,ax=axs[0,1])
plt.colorbar(im2,ax=axs[1,0])
im3.set_clim(np.min(theta_y_exact[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1]),np.max(theta_y_exact[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1]))
plt.colorbar(im3,ax=axs[1,1])
plt.suptitle("Exact vs Loc Spec Grad")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/scratch/PhaseGradLocalSpec_mu_{}.png".format(mu))
print("LocalSpec theta_x max err:", np.max(np.abs(theta_x_exact[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1]-theta_x_loc_spec)))
print("LocaSpec theta_y max err:", np.max(np.abs(theta_y_exact[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1]-theta_y_loc_spec)))



logfile.close()


