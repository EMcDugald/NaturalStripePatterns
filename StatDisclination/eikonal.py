import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
import os

Nx = 1024
Ny = 512
Lx = 80*np.pi
Ly = 40*np.pi
Rscale = .5
beta = .45
amp = .9

xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
X, Y = np.meshgrid(xx, yy)

R = Rscale*np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)
a = beta * Lx
b = beta * Ly

nmx = 256
q = 2*np.pi*np.arange(1,nmx+1,1)/nmx
imx, jmx = np.shape(X)
bdry = np.vstack((a*np.cos(q), b*np.sin(q)))
rho = np.zeros((imx,jmx))
for ii in range(imx):
    for jj in range(jmx):
        rho[ii,jj] = np.min((X[ii,jj]-bdry[0,:])**2+(Y[ii,jj]-bdry[1,:])**2)
kx = (np.pi/a)*fftfreq(jmx,1./jmx)
ky = (np.pi/b)*fftfreq(imx,1./imx)
xi, eta = np.meshgrid(kx, ky)
rho = ifft2(np.exp(-(xi**2+eta**2))*fft2(rho))
eik = np.real(amp*np.sin(np.sqrt(rho)))

fig, ax = plt.subplots()
im=ax.imshow(R,cmap='bwr')
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/"+"R.png")

fig, ax = plt.subplots()
im=ax.imshow(eik,cmap='bwr')
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/"+"eik.png")

eik_w_bdry = np.where(R>=0,eik,0)
fig, ax = plt.subplots()
im=ax.imshow(eik_w_bdry,cmap='bwr')
plt.colorbar(im)
plt.savefig(os.getcwd()+"/figs/"+"eik_w_bdry.png")