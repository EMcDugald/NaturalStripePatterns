import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
import general as sh

nx = 4096
ny = 2048
nsave = 5
tmax = 1000
dt = 1
Ly = 120*np.pi
Lx = 1.9*Ly

sh.solveSH(Lx,Ly,nx,ny,dt,tmax,nsave,"SHEllipse2to1_0124",
           Rscale=.5,beta=.45,amplitude=.1,init_flag=3,energy=True)

data = sio.loadmat(os.getcwd()+"/data/"+"SHEllipse2to1_0124.mat")
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

fig, ax = plt.subplots(nrows=2, ncols=1)
plt.subplots_adjust(wspace=.2,hspace=.2)
ax[0].imshow(U[:,:,-1],cmap='gray')
ax[1].imshow(E[:,:,-1],cmap='gray')


plt.savefig(os.getcwd()+"/figs/"+"SHEllipse2to1_0124.png")