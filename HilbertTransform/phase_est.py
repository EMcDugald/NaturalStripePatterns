import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import signal

mu = .7
dir = os.getcwd()+"/data/"
data = sio.loadmat(dir+"mu={:.3f}".format(mu)+".mat")

Ny, Nx, _ = np.shape(data['uu'])
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
pattern = data['uu'][int(7*Ny/32):int(9*Ny/32),int(Nx/4):int(3*Nx/4),10]
xx = data['xx'][int(Nx/4):int(3*Nx/4),0]
yy = data['yy'][int(7*Ny/32):int(9*Ny/32),0]
# im = ax.imshow(pattern,cmap='copper',extent=[yy[0],yy[-1],xx[0],xx[-1]])
# fig.colorbar(im, ax=ax)
# ax.set_title("pattern")
# plt.tight_layout()
# plt.show()

ny, nx = np.shape(pattern)
phase = np.zeros(shape=(ny,nx))
unwrapped_phase = np.zeros(shape=(ny,nx))
for i in range(ny):
    profile = pattern[i,:]
    hilbert = signal.hilbert(profile)
    phase[i,:] = np.arctan2(np.imag(hilbert),np.real(hilbert))
    unwrapped_phase[i,:] = np.unwrap(phase[i,:])

smooth_unwrapped_phase = np.zeros(shape=(ny,nx))
for i in range(nx):
    smooth_unwrapped_phase[:,i] = np.unwrap(unwrapped_phase[:,i])
#
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
# im = ax.imshow(phase,cmap='copper',extent=[yy[0],yy[-1],xx[0],xx[-1]])
# ax.set_title("phase")
# fig.colorbar(im, ax=ax)
# plt.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
# im = ax.imshow(unwrapped_phase,cmap='copper',extent=[yy[0],yy[-1],xx[0],xx[-1]])
# ax.set_title("unwrapped phase")
# fig.colorbar(im, ax=ax)
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
# im = ax.imshow(smooth_unwrapped_phase,cmap='copper',extent=[yy[0],yy[-1],xx[0],xx[-1]])
# ax.set_title("smooth unwrapped phase")
# fig.colorbar(im, ax=ax)
# plt.tight_layout()
# plt.show()
#
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
# im = ax.imshow(np.cos(smooth_unwrapped_phase),cmap='copper',extent=[yy[0],yy[-1],xx[0],xx[-1]])
# ax.set_title("recovered pattern")
# fig.colorbar(im, ax=ax)
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(nrows=3,ncols=1)
im1 = ax[0].imshow(smooth_unwrapped_phase,extent=[yy[0],yy[-1],xx[0],xx[-1]])
im2 = ax[1].imshow(pattern,extent=[yy[0],yy[-1],xx[0],xx[-1]])
im3 = ax[2].imshow(np.cos(smooth_unwrapped_phase),extent=[yy[0],yy[-1],xx[0],xx[-1]])
plt.colorbar(im1,ax=ax[0])
plt.colorbar(im2,ax=ax[1])
plt.colorbar(im3,ax=ax[2])
plt.suptitle("Estimated Phase, Actual Pattern, cos(phase)")
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/HilbertPhaseMu_{}.png".format(mu))



