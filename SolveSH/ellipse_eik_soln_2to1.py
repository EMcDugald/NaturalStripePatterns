import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
import general as sh

nx = 1024
ny = 512
nsave = 5
tmax = 500
dt = .1
sh.solveSH(100*np.pi,50*np.pi,nx,ny,dt,tmax,nsave,"SHEllipse2to1",
           Rscale=.5,beta=.45,amplitude=.1,init_flag=3,energy=True)

data = sio.loadmat(os.getcwd()+"/data/"+"SHEllipse2to1.mat")
U = data['uu']
E = data['ee']
x = data['xx'].T[0,:]
y = data['yy'].T[0,:]
dx = x[1]-x[0]
dy = y[1]-y[0]

x_st = round(nx/4)
x_end = x_st+2*x_st
y_st = round(ny/4)
y_end =y_st+2*y_st

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16,10))
plt.subplots_adjust(wspace=.2,hspace=.2)
ax[0,0].imshow(U[y_st:y_end,x_st:x_end,1],cmap='gray')
ax[0,1].imshow(E[y_st:y_end,x_st:x_end,1],cmap='gray')
ax[1,0].imshow(U[y_st:y_end,x_st:x_end,2],cmap='gray')
ax[1,1].imshow(E[y_st:y_end,x_st:x_end,2],cmap='gray')
ax[2,0].imshow(U[y_st:y_end,x_st:x_end,3],cmap='gray')
ax[2,1].imshow(E[y_st:y_end,x_st:x_end,3],cmap='gray')
ax[3,0].imshow(U[y_st:y_end,x_st:x_end,4],cmap='gray')
ax[3,1].imshow(E[y_st:y_end,x_st:x_end,4],cmap='gray')
ax[4,0].imshow(U[y_st:y_end,x_st:x_end,5],cmap='gray')
ax[4,1].imshow(E[y_st:y_end,x_st:x_end,5],cmap='gray')

print("total energy at time 1: ", np.sum(E[:,:,1]*dx*dy))
print("total energy at time 2: ", np.sum(E[:,:,2]*dx*dy))
print("total energy at time 3: ", np.sum(E[:,:,3]*dx*dy))
print("total energy at time 4: ", np.sum(E[:,:,4]*dx*dy))
print("total energy at time 5: ", np.sum(E[:,:,5]*dx*dy))

plt.savefig(os.getcwd()+"/figs/"+"SHEllipse2to1.png")