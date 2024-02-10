import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

Lr = 10*np.pi
La = 4*np.pi
nr = 128
na = 256
r = np.linspace(0,Lr,nr)
a = np.linspace(0,La,na)
R,A = np.meshgrid(r,a)
X0 = R*np.cos(A)
Y0 = R*np.sin(A)

### TEST 1 ###
# Z = cos(r) #
z = np.cos(R)
fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(10,20))
# top plot is 0<=a<=2pi
# bottom plot is 2pi<=a<=4pi
im0 = axs[0].scatter(X0[0:int(na/2),:],Y0[0:int(na/2),:],c=z[0:int(na/2),:])
im1 = axs[1].scatter(X0[int(na/2):na,:],Y0[int(na/2):na,:],c=z[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.show()
print("R bounds top:", R[0:int(na/2),:][0,:][0],R[0:int(na/2),:][0,:][-1])
print("A bounds top:", A[0:int(na/2),:][:,0][0],A[0:int(na/2),:][:,0][-1])
print("R bounds bottom:", R[int(na/2):na,:][0,:][0],R[int(na/2):na,:][0,:][-1])
print("A bounds bottom:", A[int(na/2):na,:][:,0][0],A[int(na/2):na,:][:,0][-1])

# first radial derivative with 2nd order finite difference #
dr = r[1]-r[0]
dzdr = (z[:,2:]-z[:,:-2])/(2*dr)
# need to redefine r to exclude r=0 and r=10pi points
r1 = R[:,1:-1][0,:]
R1,A1 = np.meshgrid(r1,a)
na1, nr1 = np.shape(R1)
X1 = R1*np.cos(A1)
Y1 = R1*np.sin(A1)
fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(10,20))
im0 = axs[0].scatter(X1[0:int(na1/2),:],Y1[0:int(na1/2),:],c=dzdr[0:int(na1/2),:])
im1 = axs[1].scatter(X1[int(na1/2):int(na1),:],Y1[int(na1/2):int(na1),:],c=dzdr[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.show()
max_err_dzdr = np.max(np.abs(-np.sin(R1)-dzdr))
mean_err_dzdr = np.mean(np.abs(-np.sin(R1)-dzdr))
print("dr:",dr)
print("theoretical err:", (dr**2)/6)
print("Max First deriv err:", max_err_dzdr)
print("Mean First deriv err:", mean_err_dzdr)
print("R bounds top:", R1[0:int(na1/2),:][0,:][0],R1[0:int(na1/2),:][0,:][-1])
print("A bounds top:", A1[0:int(na1/2),:][:,0][0],A1[0:int(na1/2),:][:,0][-1])
print("R bounds bottom:", R1[int(na1/2):na1,:][0,:][0],R1[int(na1/2):na1,:][0,:][-1])
print("A bounds bottom:", A1[int(na1/2):na1,:][:,0][0],A1[int(na1/2):na1,:][:,0][-1])


# using spectral method for first deriv in theta #
# no need to recompute grid #
kr = (2.*np.pi/Lr)*fftfreq(nr,1./nr)
ka = (2.*np.pi/La)*fftfreq(na,1./na)
Kr, Ka = np.meshgrid(kr, ka)
dzda = np.real(ifft2(1j*Ka*fft2(z)))
fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(10,20))
im0 = axs[0].scatter(X0[0:int(na/2),:],Y0[0:int(na/2),:],c=dzda[0:int(na/2),:])
im1 = axs[1].scatter(X0[int(na/2):na,:],Y0[int(na/2):na,:],c=dzda[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.show()
err_dzda = np.linalg.norm(0-dzda)
print("First deriv err:", err_dzda)

##########################################################################
