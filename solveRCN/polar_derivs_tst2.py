import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

Lr = 10*np.pi
La = 4*np.pi
nr = 512
na = 1024
r = np.linspace(0,Lr,nr)
a = np.linspace(0,La,na)
R,A = np.meshgrid(r,a)
X0 = R*np.cos(A)
Y0 = R*np.sin(A)

### TEST 2 ###
# Z = cos(ra/50) #
z = np.cos((R*A)/50)
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

# plot in polar coords
fig, ax = plt.subplots()
im0 = ax.scatter(R,A,c=z)
plt.colorbar(im0,ax=ax)
plt.show()

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
max_err_dzdr = np.max(np.abs(-(A1/50)*np.sin((A1*R1)/50)-dzdr))
mean_err_dzdr = np.mean(np.abs(-(A1/50)*np.sin((A1*R1)/50)-dzdr))
print("dr:",dr)
print("theoretical err:", ((dr**2)/6)*np.max(A1/50))
print("Max dzdr err:", max_err_dzdr)
print("Mean dzdr err:", mean_err_dzdr)
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
max_err_dzda = np.max(np.abs(-(R/50)*np.sin((A*R)/50)-dzda))
mean_err_dzda = np.mean(np.abs(-(R/50)*np.sin((A*R)/50)-dzda))
print("Max dzda err:", max_err_dzdr)
print("Mean dzda err:", mean_err_dzdr)
#
#


# second radial derivative with 2nd order finite difference #
dr = r[1]-r[0]
d2zdr2 = (z[:,2:]-2*z[:,1:-1]+z[:,:-2])/(dr**2)
# need to redefine r to exclude r=0 and r=10pi points
r2 = R[:,1:-1][0,:]
R2,A2 = np.meshgrid(r2,a)
na2, nr2 = np.shape(R2)
X2 = R2*np.cos(A2)
Y2 = R2*np.sin(A2)
fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(10,20))
im0 = axs[0].scatter(X2[0:int(na2/2),:],Y2[0:int(na2/2),:],c=d2zdr2[0:int(na2/2),:])
im1 = axs[1].scatter(X2[int(na2/2):int(na2),:],Y2[int(na2/2):int(na2),:],c=d2zdr2[int(na2/2):int(na2),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.show()
max_err_d2zdr2 = np.max(np.abs(-((A2/50)**2)*np.cos((A2*R2)/50)-d2zdr2))
mean_err_d2zdr2 = np.mean(np.abs(-((A2/50)**2)*np.cos((A2*R2)/50)-d2zdr2))
print("theoretical err:", ((dr**2)/6)*np.max((A2/500)**2))
print("Max d2zdr2 err:", max_err_d2zdr2)
print("Mean d2zdr2 err:", mean_err_d2zdr2)
print("R bounds top:", R2[0:int(na2/2),:][0,:][0],R2[0:int(na2/2),:][0,:][-1])
print("A bounds top:", A2[0:int(na2/2),:][:,0][0],A2[0:int(na2/2),:][:,0][-1])
print("R bounds bottom:", R2[int(na2/2):na2,:][0,:][0],R2[int(na2/2):na2,:][0,:][-1])
print("A bounds bottom:", A2[int(na2/2):na2,:][:,0][0],A2[int(na2/2):na2,:][:,0][-1])
#
#


# using spectral method for second deriv in theta #
# no need to recompute grid #
# ToDo: need to make the arrays compactly supported, as they are not necessarily periodic in alpha, r plane
edge_scale = .05
rl = R[0,:][0]+edge_scale*Lr
rr = R[0,:][-1]-edge_scale*Lr
al = A[:,0][0]+edge_scale*La
ar = A[:,0][-1]-edge_scale*La
smoother = (np.tanh(30*(R-rl))-np.tanh(30*(R-rr)))*(np.tanh(30*(A-al))-np.tanh(30*(A-ar)))/4
print("Min smoother:",np.min(smoother))
d2zda2 = np.real(ifft2(((1j*Ka)**2)*fft2(z*smoother)))

r3 = R[0,:][np.where((R[0,:]>rl) & (R[0,:]<rr))][20:-20]
a3 = A[:,0][np.where((A[:,0]>al) & (A[:,0]<ar))][20:-20]
R3,A3 = np.meshgrid(r3,a3)
na3, nr3 = np.shape(R3)
X3 = R3*np.cos(A3)
Y3 = R3*np.sin(A3)
row_st = np.where((A[:,0]>al) & (A[:,0]<ar))[0][20]
row_end = np.where((A[:,0]>al) & (A[:,0]<ar))[0][-20]
col_st = np.where((R[0,:]>rl) & (R[0,:]<rr))[0][20]
col_end = np.where((R[0,:]>rl) & (R[0,:]<rr))[0][-20]
d2zda2 = d2zda2[row_st:row_end,col_st:col_end]

fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(10,20))
im0 = axs[0].scatter(X3[0:int(na3/2),:],Y3[0:int(na3/2),:],c=d2zda2[0:int(na3/2),:])
im1 = axs[1].scatter(X3[int(na3/2):na3,:],Y3[int(na3/2):na3,:],c=d2zda2[int(na3/2):na3,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.show()
max_err_d2zda2 = np.max(np.abs(-((R3/50)**2)*np.cos((A3*R3)/50)-d2zda2))
mean_err_d2zda2 = np.mean(np.abs(-((R3/50)**2)*np.cos((A3*R3)/50)-d2zda2))
print("Max d2zda2 err:", max_err_d2zda2)
print("Mean d2zda2 err:", mean_err_d2zda2)



#
#
# # mixed partial- first in r then in alpha #
# dr = r[1]-r[0]
# dzdr = (z[:,2:]-z[:,:-2])/(2*dr)
# # need to redefine r to exclude r=0 and r=10pi points
# r1 = R[:,1:-1][0,:]
# # compute angular derivative now
# R1,A1 = np.meshgrid(r1,a)
# na1, nr1 = np.shape(R1)
# Lr1 = R1[0,:][-1]-R1[0,:][0]
# La1 = A1[:,0][-1]-A1[:,0][0]
# kr1 = (2.*np.pi/Lr1)*fftfreq(nr1,1./nr1)
# ka1 = (2.*np.pi/La1)*fftfreq(na1,1./na1)
# Kr1, Ka1 = np.meshgrid(kr1, ka1)
# d2zdrda = np.real(ifft2((1j*Ka1)*fft2(dzdr)))
# X1 = R1*np.cos(A1)
# Y1 = R1*np.sin(A1)
# fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(10,20))
# im0 = axs[0].scatter(X1[0:int(na1/2),:],Y1[0:int(na1/2),:],c=d2zdrda[0:int(na1/2),:])
# im1 = axs[1].scatter(X1[int(na1/2):int(na1),:],Y1[int(na1/2):int(na1),:],c=d2zdrda[int(na1/2):int(na1),:])
# plt.colorbar(im0,ax=axs[0])
# plt.colorbar(im1,ax=axs[1])
# plt.show()
# max_err_d2zdrda = np.max(np.abs(-A1*R1*np.cos(A1*R1)-d2zdrda))
# mean_err_d2zdrda = np.mean(np.abs(-A1*R1*np.cos(A1*R1)-d2zdrda))
# print("Max d2zdrda err:", max_err_dzdr)
# print("Mean d2zdrda err:", mean_err_dzdr)
#
#
# # mixed partial, first in angular, then in radial
# kr = (2.*np.pi/Lr)*fftfreq(nr,1./nr)
# ka = (2.*np.pi/La)*fftfreq(na,1./na)
# Kr, Ka = np.meshgrid(kr, ka)
# dzda = np.real(ifft2(1j*Ka*fft2(z)))
# dr = r[1]-r[0]
# d2zdadr = (dzda[:,2:]-dzda[:,:-2])/(2*dr)
# r1 = R[:,1:-1][0,:]
# R1,A1 = np.meshgrid(r1,a)
# na1, nr1 = np.shape(R1)
# X1 = R1*np.cos(A1)
# Y1 = R1*np.sin(A1)
# fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(10,20))
# im0 = axs[0].scatter(X1[0:int(na1/2),:],Y1[0:int(na1/2),:],c=d2zdadr[0:int(na1/2),:])
# im1 = axs[1].scatter(X1[int(na1/2):int(na1),:],Y1[int(na1/2):int(na1),:],c=d2zdadr[int(na1/2):int(na1),:])
# plt.colorbar(im0,ax=axs[0])
# plt.colorbar(im1,ax=axs[1])
# plt.show()
# max_err_d2zdadr = np.max(np.abs(-R1*A1*np.cos(A1*R1)-d2zdadr))
# mean_err_d2zdadr = np.mean(np.abs(-R1*A1*np.cos(A1*R1)-d2zdadr))
# print("Max dzdr err:", max_err_d2zdadr)
# print("Mean dzdr err:", mean_err_d2zdadr)
#
#
#
#
# ##########################################################################