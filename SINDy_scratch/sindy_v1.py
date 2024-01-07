import numpy as np
from wave_dirs_v1 import get_wave_dirs
from wave_lens_v1 import get_wave_lens_global
from utils import snaking_indices
import sys
import os
import matplotlib.pyplot as plt
from fit_pde import TrainSTRidge, print_pde
from derivatives import FiniteDiffDerivs

######### MAKE THE WAVE VECTOR FIELD ##################

mu = .7

f = open(os.getcwd()+"/logs/sindy_v1_mu_{}.out".format(mu), 'w')
sys.stdout = f

Nx = 2048
Ny = 1024
k1 = np.sqrt(1-mu**2)
k2 = mu
Ly = 10*2*np.pi/k1
Lx = 2*Ly
xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
X,Y = np.meshgrid(xx,yy)
theta = k1*X + np.log(2*np.cosh(k2*Y))
W = np.cos(theta)
wave_vec_aspect = [.5,.5]  # sets dimension of sampling region. in this case, a rectangle of dim .5Ny, .5Nx, centered at 0
wave_vec_subsample = 8

wave_dirs = get_wave_dirs(mu,Nx,Ny,X,Y,W,
                          wave_vec_aspect,wave_vec_subsample,
                          save=False)

max_wl = 12.0
wl = get_wave_lens_global(mu,max_wl,Lx,Ly,Nx,Ny,X,Y,W,wave_dirs,wave_vec_aspect,
                             wave_vec_subsample,save=False)

wn = (2*np.pi/wl)

def kx(x,y):
    return k1*np.ones(shape=np.shape(x))

def ky(x,y):
    return k2*np.tanh(k2*y)

kx_exact = kx(X,Y)
ky_exact = ky(X,Y)
wave_nums = np.sqrt(kx_exact**2+ky_exact**2)

rect_y = int(wave_vec_aspect[0]*Ny)
rect_x = int(wave_vec_aspect[1]*Nx)
indices = snaking_indices(rect_y, rect_x, wave_vec_subsample)
rect_y_ss = int(rect_y/wave_vec_subsample)
rect_x_ss = int(rect_x/wave_vec_subsample)
wave_nums_exact = np.zeros(shape=(rect_y_ss, rect_x_ss))
y_shift = int(.5*(Ny-rect_y))
x_shift = int(.5*(Nx-rect_x))
for pair in indices:
    r, c = pair
    wave_nums_exact[int(r / wave_vec_subsample), int(c / wave_vec_subsample)] += wave_nums[r + y_shift,c + x_shift]

mask = np.where(wl==np.inf,1,0)
masked_wn = np.where(mask==0,wn,0)
masked_exact = np.where(mask==0,wave_nums_exact,0)

print("wn arr size:",np.shape(wl))
print("total masked:",np.sum(mask))

print("l2 rel error estimated v exact:",np.linalg.norm(masked_wn-masked_exact)/np.linalg.norm(masked_exact))
print("mean abs error estimated v exact:",np.mean(np.abs(masked_wn-masked_exact)))
print("max abs error estimated v exact:",np.max(np.abs(masked_wn-masked_exact)))
print("min abs error estimated v exact:",np.min(np.abs(masked_wn-masked_exact)))

print("max abs rel error estimated v exact:",np.max(np.abs(masked_wn-masked_exact)/np.abs(masked_exact)))

print("max estimated wave num:",np.max(masked_wn))
print("min estimated wave num:",np.min(masked_wn))
print("max possible wave num:",np.max(wave_nums_exact))
print("min possible wave num:",np.min(wave_nums_exact))

rect_y = int(wave_vec_aspect[0]*Ny)
rect_x = int(wave_vec_aspect[1]*Nx)
indices = snaking_indices(rect_y, rect_x, wave_vec_subsample)
x_coarse = np.asarray([X[i+y_shift,j+x_shift] for i,j in indices])
y_coarse = np.asarray([Y[i+y_shift,j+x_shift] for i,j in indices])

fig,ax = plt.subplots()
im = ax.scatter(x_coarse,y_coarse,c=masked_wn.flatten())
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/debug/"+"approx_wavenums_mu_{}.png".format(mu))

fig,ax = plt.subplots()
im = ax.scatter(x_coarse,y_coarse,c=masked_exact.flatten())
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/debug/"+"exact_wavenums_mu_{}.png".format(mu))

kx_approx = np.cos(wave_dirs)*(masked_wn.reshape(np.shape(wave_dirs)))
ky_approx = np.sin(wave_dirs)*(masked_wn.reshape(np.shape(wave_dirs)))
rect_y = int(wave_vec_aspect[0]*Ny)
rect_x = int(wave_vec_aspect[1]*Nx)
indices = snaking_indices(rect_y, rect_x, wave_vec_subsample)
rect_y_ss = int(rect_y/wave_vec_subsample)
rect_x_ss = int(rect_x/wave_vec_subsample)
kx_exact_ss = np.zeros(shape=(rect_y_ss, rect_x_ss))
ky_exact_ss = np.zeros(shape=(rect_y_ss, rect_x_ss))
y_shift = int(.5*(Ny-rect_y))
x_shift = int(.5*(Nx-rect_x))
for pair in indices:
    r, c = pair
    kx_exact_ss[int(r / wave_vec_subsample), int(c / wave_vec_subsample)] += kx_exact[r + y_shift,c + x_shift]
    ky_exact_ss[int(r / wave_vec_subsample), int(c / wave_vec_subsample)] += ky_exact[r + y_shift,c + x_shift]

kx_approx = np.abs(kx_approx)
ky_approx = (np.where(y_coarse.flatten()>0,np.abs(ky_approx).flatten(),-np.abs(ky_approx).flatten())).reshape(np.shape(kx_approx))

fig,ax = plt.subplots()
im = ax.scatter(x_coarse,y_coarse,c=kx_exact_ss.flatten())
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/debug/"+"exact_kx_ss_mu_{}.png".format(mu))

fig,ax = plt.subplots()
im = ax.scatter(x_coarse,y_coarse,c=ky_exact_ss.flatten())
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/debug/"+"exact_ky_ss_mu_{}.png".format(mu))

fig,ax = plt.subplots()
im = ax.scatter(x_coarse,y_coarse,c=kx_approx.flatten())
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/debug/"+"approx_kx_mu_{}.png".format(mu))

fig,ax = plt.subplots()
im = ax.scatter(x_coarse,y_coarse,c=ky_approx.flatten())
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/debug/"+"approx_ky_mu_{}.png".format(mu))


print("l2 rel error estimated v exact kx:",np.linalg.norm(kx_approx-kx_exact_ss)/np.linalg.norm(kx_exact_ss))
print("l2 rel error estimated v exact ky:",np.linalg.norm(ky_approx-ky_exact_ss)/np.linalg.norm(ky_exact_ss))


#f.close()




############### SINDY PROCEDURE HERE ###################

rect_y_ss = int(rect_y / wave_vec_subsample)
rect_x_ss = int(rect_x / wave_vec_subsample)
x_coarse_vals = np.sort(x_coarse)[::rect_y_ss]
y_coarse_vals = np.sort(y_coarse)[::rect_x_ss]
dx = x_coarse_vals[1] - x_coarse_vals[0]
dy = y_coarse_vals[1] - y_coarse_vals[0]

sampled_xs = x_coarse_vals
sampled_ys = y_coarse_vals
kxs = kx_approx
kys = ky_approx

x_idxs = np.arange(0,len(sampled_xs)-2,1)
y_idxs = np.arange(0,len(sampled_ys)-2,1)
num_points = len(x_idxs)*len(y_idxs) #subtracting 2 due to spatial derivative calculations
print("feature vec length: ",num_points)

ones = np.ones((num_points,1))
norm_ks = np.zeros((num_points,1))
div_ks = np.zeros((num_points,1))
cos_ys = np.zeros((num_points,1))
sin_xs = np.zeros((num_points,1))
x_cubes = np.zeros((num_points,1))
xs = np.zeros((num_points,1))
ys = np.zeros((num_points,1))

i=0
sampled_x = sampled_xs
sampled_y = sampled_ys
sampled_X, sampled_Y = np.meshgrid(sampled_x,sampled_y)
norm_k = np.sqrt(kxs ** 2 + kys ** 2)[1:-1,1:-1]
print("average value of norm_k:",norm_k.mean())
dkx_dx = FiniteDiffDerivs(kxs,dx,dy,type='x')
print("average value of derivative wrt x of kx:",dkx_dx.mean())
print("abs max value of derivative wrt x of kx:", np.max(np.abs(dkx_dx)))
dky_dy = FiniteDiffDerivs(kys,dx,dy,type='y')
div_k = dkx_dx+dky_dy
cos_y = np.cos(sampled_Y[1:-1,1:-1])
sin_x = np.sin(sampled_X[1:-1,1:-1])
x_cube = (sampled_X**3)[1:-1,1:-1]
x = sampled_X[1:-1]
y = sampled_Y[1:-1]
for x_idx in x_idxs:
    for y_idx in y_idxs:
        norm_ks[i] = norm_k[y_idx,x_idx]
        div_ks[i] = div_k[y_idx,x_idx]
        cos_ys[i] = cos_y[y_idx,x_idx]
        sin_xs[i] = sin_x[y_idx,x_idx]
        x_cubes[i] = x_cube[y_idx,x_idx]
        xs[i] = x[y_idx,x_idx]
        ys[i] = y[y_idx,x_idx]
        i+=1

X1 = np.hstack([norm_ks**2, norm_ks**4, np.exp(ys),
                   cos_ys, sin_xs, x_cubes,div_ks, ys**2, np.arctan(xs*ys)])
description1 = ['norm_ks_sq','norm_ks_fth', 'e^y',
               'cos_y','sin_x','x_cb','div_k','y^2','atan(xy)']

c1 = TrainSTRidge(X1,div_ks**2-1,10**-5,1)

print("c1=",c1)

print_pde(c1, description1)


X2 = np.hstack([np.exp(ys),ys**2,np.arctan(xs*ys),
                   cos_ys,sin_xs, x_cubes,div_ks,div_ks**2])
description2 = ['e^y','y^2','atan(xy)',
               'cos_y','sin_x','x_cb','div_k','div_ks_sq']

c2 = TrainSTRidge(X2,1-2*norm_ks**2+norm_ks**4,10**-5,1)

print("c2=",c2)

print_pde(c2, description2)

X3 = np.hstack([np.exp(ys),ys**2,np.arctan(xs*ys),
                norm_ks**4,ones,cos_ys,sin_xs, x_cubes])
description3 = ['e^y','y^2','atan(xy)','norm_ks_frth','',
               'cos_y','sin_x','x_cb']

c3 = TrainSTRidge(X3,1-2*norm_ks**2-div_ks**2,10**-5,1)

print("c3=",c3)

print_pde(c3, description3)


X4 = np.hstack([np.exp(ys),ys**2,np.arctan(xs*ys),cos_ys,sin_xs, x_cubes])
description4 = ['e^y','y^2','atan(xy)',
               'cos_y','sin_x','x_cb']

c4 = TrainSTRidge(X4,div_ks**2 + 1 -2*norm_ks**2 + norm_ks**4,10**-5,1)

print("c4=",c4)

print_pde(c4, description4)

f.close()