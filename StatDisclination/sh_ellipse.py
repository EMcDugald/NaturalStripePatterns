import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os
import sh_general as sh

nx = 1024
ny = 512
nsave = 500
tmax = 250
dt = .5
sh.solveSH(60*np.pi,30*np.pi,nx,ny,dt,tmax,nsave,"SHEllipse",
           Rscale=.5,beta=.45,amplitude=.3,init_flag=3,energy=True)

data = sio.loadmat(os.getcwd()+"/data/"+"SHEllipse.mat")
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

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16,20))
ax[0,0].imshow(U[y_st:y_end,x_st:x_end,1],cmap='bwr')
ax[0,1].imshow(E[y_st:y_end,x_st:x_end,1],cmap='bwr')
ax[1,0].imshow(U[y_st:y_end,x_st:x_end,10],cmap='bwr')
ax[1,1].imshow(E[y_st:y_end,x_st:x_end,10],cmap='bwr')
ax[2,0].imshow(U[y_st:y_end,x_st:x_end,50],cmap='bwr')
ax[2,1].imshow(E[y_st:y_end,x_st:x_end,50],cmap='bwr')
ax[3,0].imshow(U[y_st:y_end,x_st:x_end,250],cmap='bwr')
ax[3,1].imshow(E[y_st:y_end,x_st:x_end,250],cmap='bwr')
ax[4,0].imshow(U[y_st:y_end,x_st:x_end,499],cmap='bwr')
ax[4,1].imshow(E[y_st:y_end,x_st:x_end,499],cmap='bwr')
plt.savefig(os.getcwd()+"/figs/"+"SHEllipse.png")

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16,20))
ax[0,0].imshow(U[:,:,1],cmap='bwr')
ax[0,1].imshow(E[:,:,1],cmap='bwr')
ax[1,0].imshow(U[:,:,10],cmap='bwr')
ax[1,1].imshow(E[:,:,10],cmap='bwr')
ax[2,0].imshow(U[:,:,50],cmap='bwr')
ax[2,1].imshow(E[:,:,50],cmap='bwr')
ax[3,0].imshow(U[:,:,250],cmap='bwr')
ax[3,1].imshow(E[:,:,250],cmap='bwr')
ax[4,0].imshow(U[:,:,499],cmap='bwr')
ax[4,1].imshow(E[:,:,499],cmap='bwr')

plt.savefig(os.getcwd()+"/figs/"+"SHEllipseFull.png")