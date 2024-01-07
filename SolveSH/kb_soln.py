import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
import knee_bend as sh

nx = 1536
ny = 768
nsave = 5
tmax = 200
dt = .05
sh.solveSH(110*np.pi,55*np.pi,nx,ny,dt,tmax,.3,'SHKB1',xlim_scale=.9,nsave=nsave,tanh_scale=5.0)

data = sio.loadmat(os.getcwd()+"/data/"+"SHKB1.mat")

t = data['tt'].T[:,0]
x = data['xx'].T[0,:]
y = data['yy'].T[0,:]
U = data['uu']
e = data['ee']
dx = x[1]-x[0]
dy = y[1]-y[0]

x_st = round(nx/3)
x_end = 2*x_st
y_st = round(ny/3)
y_end = 2*y_st

print("interior data shape:", np.shape(U[0,y_st:y_end,x_st:x_end]))

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(12,15))
axs[0,0].imshow(U[1,y_st:y_end,x_st:x_end],cmap='gray')
axs[0,1].imshow(e[1,y_st:y_end,x_st:x_end],cmap='gray')
axs[1,0].imshow(U[2,y_st:y_end,x_st:x_end],cmap='gray')
axs[1,1].imshow(e[2,y_st:y_end,x_st:x_end],cmap='gray')
axs[2,0].imshow(U[3,y_st:y_end,x_st:x_end],cmap='gray')
axs[2,1].imshow(e[3,y_st:y_end,x_st:x_end],cmap='gray')
axs[3,0].imshow(U[4,y_st:y_end,x_st:x_end],cmap='gray')
axs[3,1].imshow(e[4,y_st:y_end,x_st:x_end],cmap='gray')
axs[4,0].imshow(U[5,y_st:y_end,x_st:x_end],cmap='gray')
axs[4,1].imshow(e[5,y_st:y_end,x_st:x_end],cmap='gray')


print("total energy at time 1: ", np.sum(e[1,:,:]*dx*dy))
print("total energy at time 1: ", np.sum(e[2,:,:]*dx*dy))
print("total energy at time 1: ", np.sum(e[3,:,:]*dx*dy))
print("total energy at time 1: ", np.sum(e[4,:,:]*dx*dy))
print("total energy at time 1: ", np.sum(e[5,:,:]*dx*dy))

plt.savefig(os.getcwd()+"/figs/"+"SHKB1.png")
