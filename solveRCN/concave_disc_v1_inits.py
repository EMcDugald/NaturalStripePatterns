import numpy as np
import matplotlib.pyplot as plt


Lx = 20*np.pi
Ly = 20*np.pi
Nx = 256
Ny = 256
x = np.linspace(-Lx/2,Lx/2,Nx)
y = np.linspace(-Ly/2,Ly/2,Ny)
X,Y = np.meshgrid(x,y)
Rscale = .5
beta = .49




### SET 1 ###
angle_fctr = 2.25
theta = (2./angle_fctr)*(np.sqrt(X**2+Y**2))**(angle_fctr/2.)*np.sin((angle_fctr/2.)*np.arctan2(X,Y))
fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(5,12))
im1 = ax[0].scatter(X,Y,c=theta,cmap='bwr',alpha=.1)
plt.colorbar(im1,ax=ax[0])
im2 = ax[1].scatter(X,Y,c=np.cos(theta),cmap='bwr',alpha=.1)
plt.colorbar(im2,ax=ax[1])
R = Rscale*(np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1)
im3 = ax[2].scatter(X,Y,c=np.cos(theta)*R,cmap='bwr',alpha=.1)
plt.colorbar(im3, ax=ax[2])
plt.show()


### SET 2 ###
angle_fctr = 2.5
theta = (2./angle_fctr)*(np.sqrt(X**2+Y**2))**(angle_fctr/2.)*np.sin((angle_fctr/2.)*np.arctan2(X,Y))
fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(5,12))
im1 = ax[0].scatter(X,Y,c=theta,cmap='bwr',alpha=.1)
plt.colorbar(im1,ax=ax[0])
im2 = ax[1].scatter(X,Y,c=np.cos(theta),cmap='bwr',alpha=.1)
plt.colorbar(im2,ax=ax[1])
R = Rscale*(np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1)
im3 = ax[2].scatter(X,Y,c=np.cos(theta)*R,cmap='bwr',alpha=.1)
plt.colorbar(im3, ax=ax[2])
plt.show()

### SET 3 ###
angle_fctr = 2.75
theta = (2./angle_fctr)*(np.sqrt(X**2+Y**2))**(angle_fctr/2.)*np.sin((angle_fctr/2.)*np.arctan2(X,Y))
fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(5,12))
im1 = ax[0].scatter(X,Y,c=theta,cmap='bwr',alpha=.1)
plt.colorbar(im1,ax=ax[0])
im2 = ax[1].scatter(X,Y,c=np.cos(theta),cmap='bwr',alpha=.1)
plt.colorbar(im2,ax=ax[1])
R = Rscale*(np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1)
im3 = ax[2].scatter(X,Y,c=np.cos(theta)*R,cmap='bwr',alpha=.1)
plt.colorbar(im3, ax=ax[2])
plt.show()


### SET 4 ###
angle_fctr = 3.0
theta = (2./angle_fctr)*(np.sqrt(X**2+Y**2))**(angle_fctr/2.)*np.sin((angle_fctr/2.)*np.arctan2(X,Y))
fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(5,12))
im1 = ax[0].scatter(X,Y,c=theta,cmap='bwr',alpha=.1)
plt.colorbar(im1,ax=ax[0])
im2 = ax[1].scatter(X,Y,c=np.cos(theta),cmap='bwr',alpha=.1)
plt.colorbar(im2,ax=ax[1])
R = Rscale*(np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1)
im3 = ax[2].scatter(X,Y,c=np.cos(theta)*R,cmap='bwr',alpha=.1)
plt.colorbar(im3, ax=ax[2])
plt.show()




### POLAR COORD STUFF ###
angle_fctr = 3 #default is 3 for equal angles
r = np.linspace(0,np.sqrt(50*np.pi),128)
alpha = np.linspace(0,2*np.pi,256)
R, Alpha = np.meshgrid(r,alpha)
theta = (2./angle_fctr)*(R**(angle_fctr/2.))*np.sin((angle_fctr/2.)*Alpha)
X = R*np.cos(Alpha)
Y = R*np.sin(Alpha)
fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(5,8))
im1 = ax[0].scatter(X,Y,c=theta,cmap='bwr',alpha=.1)
plt.colorbar(im1,ax=ax[0])
im2 = ax[1].scatter(X,Y,c=np.cos(theta),cmap='bwr',alpha=.1)
plt.show()
