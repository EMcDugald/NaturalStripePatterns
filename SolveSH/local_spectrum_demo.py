import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, fftshift
import os
import matplotlib.pyplot as plt



mu = .3
#mu = .5
#mu = .7
#mu = .9

Nx = 512
Ny = 256
k1 = np.sqrt(1-mu**2)
k2 = mu
Ly = 20*np.pi/k1
Lx = 2*Ly
xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
X,Y = np.meshgrid(xx,yy)
theta = k1*X + np.log(2*np.cosh(k2*Y))
W = np.cos(theta)

kx = (2. * np.pi / Lx) * fftfreq(Nx, 1. / Nx)  # wave numbers
ky = (2. * np.pi / Ly) * fftfreq(Ny, 1. / Ny)
Kx, Ky = np.meshgrid(kx, ky)

col_indices = np.where((X[0,:]>=-Lx/4) & (X[0,:]<Lx/4))[0]
row_indices = np.where((Y[:,0]>=-Ly/4) & (Y[:,0]<Ly/4))[0]
innerW = W[row_indices[0]:row_indices[-1],col_indices[0]:col_indices[-1]]
innerX = X[row_indices[0]:row_indices[-1],col_indices[0]:col_indices[-1]]
innerY = Y[row_indices[0]:row_indices[-1],col_indices[0]:col_indices[-1]]

def Gaussian(x0,y0,X,Y,sigma):
    exponent = (X-x0)**2 + (Y-y0)**2
    return np.exp(-exponent/(sigma**2))

G = Gaussian(innerX[0,0],innerY[0,0],X,Y,3)

fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(12,14))
im1 = ax[0].scatter(X.flatten(),Y.flatten(),c=W.flatten(),cmap='bwr')
im2 = ax[1].scatter(X.flatten(),Y.flatten(),c=(W*G).flatten(),cmap='bwr')
im3 = ax[2].scatter(Kx.flatten(),Ky.flatten(),c=np.abs(fft2(W*G)).flatten(),cmap='bwr')
ax[0].title.set_text('Pattern')
ax[1].title.set_text('Pattern times Gaussian')
ax[2].title.set_text('Abs Spectrum')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/local_spec/"+"ls1_scatter.png")
plt.close()

fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(12,14))
im1 = ax[0].imshow(W,cmap='bwr')
im2 = ax[1].imshow((W*G),cmap='bwr')
im3 = ax[2].imshow(np.abs(fftshift(fft2(W*G))),cmap='bwr')
ax[0].title.set_text('Pattern')
ax[1].title.set_text('Pattern times Gaussian')
ax[2].title.set_text('Abs Spectrum')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/local_spec/"+"ls1_imshow.png")
plt.close()

print("debug")

f = G*W
spec = fftshift(fft2(f))
indices = np.argsort(-np.abs(spec).flatten())
Kxs = fftshift(Kx).flatten()[indices]
Kys = fftshift(Ky).flatten()[indices]
FourierCoeffs = spec.flatten()[indices]
#dominant_wave = np.real((FourierCoeffs[0]*np.exp(-1j*(Kxs[0]*X+Kys[0]*Y)))/(Nx*Ny))
dominant_wave = np.real(np.exp(1j*(Kxs[0]*X+Kys[0]*Y)))
fig, ax = plt.subplots(nrows=2,ncols=1)
im1 = ax[0].imshow(f,cmap='bwr')
im2 = ax[1].imshow(dominant_wave*G,cmap='bwr')
ax[0].title.set_text('Local Pattern')
ax[1].title.set_text('Plane Wave')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/local_spec/"+"dominant_pw1.png")
plt.close()


