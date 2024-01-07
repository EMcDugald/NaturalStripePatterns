import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import special
from derivatives import FiniteDiffDerivs

###  FULL PATTERN  ###

kb = 1.0
beta = 0.1
Lx = 30*np.pi
Ly = Lx
Nx = 1024
Ny = 1024
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
xx = np.arange(-Lx/2,Lx/2+dx/2,dx)
yy = np.arange(-Ly/2,Ly/2+dy/2,dy)
X,Y = np.meshgrid(xx,yy)
theta = kb*X + (1/beta)*np.log(.5*(1+np.exp(beta*np.pi*np.sign(X)))+
                               .5*(1-np.exp(beta*np.pi*np.sign(X)))*
                               special.erf(np.sqrt(beta*kb)*Y/np.sqrt(np.abs(X))))

W = np.cos(theta)
fig, ax = plt.subplots()
im=ax.imshow(W,cmap='bwr')
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/"+"pattern.png")

###  Dislocation Derivatives  ###

def grad_theta(X,Y,kb,beta):
    exp_beta = np.where(X<0,np.exp(-beta*np.pi),np.exp(beta*np.pi))
    exp = (np.sqrt(beta*kb)*Y)/np.sqrt(np.abs(X))
    erf = special.erf(exp)
    rat = 1./(.5*(1.+exp_beta)+.5*(1.-exp_beta)*erf)
    derf = (2./np.sqrt(np.pi))*np.exp(-(exp)**2)
    dexp = (-X)/(2*(np.abs(X))**(5./2.))
    dthetadx = kb + (1./beta)*rat*.5*(1-exp_beta)*derf*np.sqrt(beta*kb)*Y*dexp
    dthetady = (1./beta)*rat*.5*(1-exp_beta)*derf*(np.sqrt(beta*kb)/np.sqrt(np.abs(X)))
    return dthetadx, dthetady

kx, ky = grad_theta(X,Y,kb,beta)

fig, ax = plt.subplots()
im=ax.imshow(kx,cmap='bwr')
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/"+"kx.png")

fig, ax = plt.subplots()
im=ax.imshow(ky,cmap='bwr')
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/"+"ky.png")

kx_dir = kx/np.sqrt(kx**2+ky**2)
ky_dir = ky/np.sqrt(kx**2+ky**2)
fig, ax = plt.subplots()
ax.quiver(X[::32,::32].flatten(),Y[::32,::32].flatten(),kx_dir[::32,::32].flatten(),ky_dir[::32,::32].flatten())
plt.savefig(os.getcwd()+"/figs/"+"quiver.png")

fig, ax = plt.subplots()
im=ax.imshow(np.sqrt(kx**2+ky**2),cmap='bwr',clim=[0,2])
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/"+"wn.png")

phi = np.zeros((Ny,Nx))
dx = X[0,1]-X[0,0]
dy = Y[1,0]-Y[0,0]
kx_fd = FiniteDiffDerivs(theta,dx,dy,type='x')
ky_fd = FiniteDiffDerivs(theta,dx,dy,type='y')

fig, ax = plt.subplots()
im=ax.imshow(kx_fd,cmap='bwr')
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/"+"kx_fd.png")

fig, ax = plt.subplots()
im=ax.imshow(ky_fd,cmap='bwr')
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/"+"ky_fd.png")


