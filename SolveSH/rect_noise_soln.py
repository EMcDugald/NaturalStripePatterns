import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
import general as sh

nsave = 10
sh.solveSH(20*np.pi,10*np.pi,512,256,.5,500,nsave,"SHRect1",
           Rscale=.5,beta=.45,amplitude=.1,init_flag=1,energy=True)

data = sio.loadmat(os.getcwd()+"/data/"+"SHRect1.mat")
U = data['uu']
E = data['ee']
x = data['xx'].T[0,:]
y = data['yy'].T[0,:]
dx = x[1]-x[0]
dy = y[1]-y[0]

fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
plt.subplots_adjust(wspace=.2,hspace=.2)
ax[0,0].imshow(U[:,:,0],cmap='gray')
ax[0,1].imshow(E[:,:,0],cmap='gray')
ax[1,0].imshow(U[:,:,2],cmap='gray')
ax[1,1].imshow(E[:,:,2],cmap='gray')
ax[2,0].imshow(U[:,:,4],cmap='gray')
ax[2,1].imshow(E[:,:,4],cmap='gray')
ax[3,0].imshow(U[:,:,6],cmap='gray')
ax[3,1].imshow(E[:,:,6],cmap='gray')
ax[4,0].imshow(U[:,:,9],cmap='gray')
ax[4,1].imshow(E[:,:,9],cmap='gray')

print("total energy at time 0: ", np.sum(E[:,:,0]*dx*dy))
print("total energy at time 50: ", np.sum(E[:,:,2]*dx*dy))
print("total energy at time 100: ", np.sum(E[:,:,4]*dx*dy))
print("total energy at time 150: ", np.sum(E[:,:,6]*dx*dy))
print("total energy at time 200: ", np.sum(E[:,:,9]*dx*dy))
plt.savefig(os.getcwd()+"/figs/"+"SHRect1.png")