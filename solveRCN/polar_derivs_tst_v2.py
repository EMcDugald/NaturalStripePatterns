import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

Lr = 10*np.pi
La = 4*np.pi
nr = 256
na = 512
r = np.linspace(0,Lr-Lr/nr,nr)
a = np.linspace(0,La-La/na,na)
R,A = np.meshgrid(r,a)
X = R*np.cos(A)
Y = R*np.sin(A)

### PLOT THE SURFACE ###
theta = ((2./3.)*R**(3./2.))*np.sin((3./2.)*A)
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X[0:int(na/2),:],Y[0:int(na/2),:],c=theta[0:int(na/2),:])
im1 = axs[1].scatter(X[int(na/2):na,:],Y[int(na/2):na,:],c=theta[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Phase Surface: Cartesian View")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R[0:int(na/2),:],A[0:int(na/2),:],c=theta[0:int(na/2),:])
im1 = axs[1].scatter(R[int(na/2):na,:],A[int(na/2):na,:],c=theta[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Phase Surface: Polar View")
plt.tight_layout()
plt.show()

### COMPUTE FIRST RADIAL DERIVATIVE ###
dr = r[1]-r[0]
dthetadr_centered = (-theta[:,4:]+8*theta[:,3:-1]-8*theta[:,1:-3]+theta[:,:-4])/(12*dr)
dthetadr_forward = -(25/12)*theta[:,:-4] + 4*theta[:,1:-3]-3*theta[:,2:-2]+(4/3)*theta[:,3:-1]-(1/4)*theta[:,4:]
#dthetadr_backward = (1/4)*theta[:,4:]-(4/3)*theta[:,3:-1]+3*theta[:,2:-2]-4*theta[:,1:-3]+(25/12)*theta[:,:-4]
#dthetadr_backward = (1/4)*theta[:,:-4]-(4/3)*theta[:,1:-3]+3*theta[:,2:-2]-4*theta[:,3:-1]+(25/12)*theta[:,4:]
dthetadr_backward = (25/12)*theta[:,4:]-4*theta[:,3:-1]+3*theta[:,2:-2]-(4/3)*theta[:,1:-3]+(1/4)*theta[:,:-4]
dtheta_dr = np.zeros(np.shape(R))
dtheta_dr[:,0:2] = dthetadr_forward[:,0:2]
dtheta_dr[:,2:-2] = dthetadr_centered
dtheta_dr[:,[-2,-1]] = dthetadr_backward[:,[-2,-1]]
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X[0:int(na/2),:],Y[0:int(na/2),:],c=dtheta_dr[0:int(na/2),:])
im1 = axs[1].scatter(X[int(na/2):int(na),:],Y[int(na/2):int(na),:],c=dtheta_dr[int(na/2):int(na),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("dfdr Cartesian")
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R[0:int(na/2),:],A[0:int(na/2),:],c=dtheta_dr[0:int(na/2),:])
im1 = axs[1].scatter(R[int(na/2):int(na),:],A[int(na/2):int(na),:],c=dtheta_dr[int(na/2):int(na),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("dfdr Polar")
plt.tight_layout()
plt.show()
dthetadr_exact = np.sqrt(R)*np.sin(3*A/2)
max_dthetadr_err = np.max(np.abs(dtheta_dr-dthetadr_exact))
mean_dthetadr_err = np.mean(np.abs(dtheta_dr-dthetadr_exact))
L2_dthetadr_err = np.linalg.norm(dtheta_dr-dthetadr_exact)
print("radial grid spacing (and 4th power):",dr, dr**4)
print("max_dthetadr_err:",max_dthetadr_err)
print("mean_dthetadr_err:",mean_dthetadr_err)
print("L2_dthetadr_err:",L2_dthetadr_err)

fig, ax = plt.subplots()
im = ax.imshow(np.abs(dthetadr_exact-dtheta_dr))
plt.colorbar(im,ax=ax)
plt.show()