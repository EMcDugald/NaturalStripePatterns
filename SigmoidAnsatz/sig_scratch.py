import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

Lx = 50*np.pi
Ly = 50*np.pi
Nx = 256
Ny = 256
xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
X, Y = np.meshgrid(xx, yy)

def sigmoid(x,y,a,b,c,d,e,f):
    return c/(1+np.exp(f*(a*(x-d)+b*(y-e))))

phi1 = sigmoid(X,Y,1,1,75,0,0,.01)
phi2 = sigmoid(X,Y,-1,1,75,0,0,.01)


theta = np.log(np.exp(phi1)+np.exp(phi2))
pattern = np.cos(theta)
fig, ax = plt.subplots()
im = ax.imshow(pattern)
plt.colorbar(im,ax=ax)
plt.show()


phi1 = sigmoid(X,Y,1,1,75,0,0,.01)
phi2 = sigmoid(X,Y,-1,1,75,0,0,.01)
phi3 = sigmoid(X,Y,-1,-1,75,0,0,.01)
phi4 = sigmoid(X,Y,1,-1,75,0,0,.01)

theta = np.log(np.exp(phi1)+np.exp(phi2)+np.exp(phi3)+np.exp(phi4))
pattern = np.cos(theta)
fig, ax = plt.subplots()
im = ax.imshow(pattern)
plt.colorbar(im,ax=ax)
plt.show()

phi1 = .9*sigmoid(X,Y,1,1,75,0,0,.01)
phi2 = .9*sigmoid(X,Y,-1,1,75,0,0,.01)
phi3 = -3*sigmoid(X,Y,-1,-1,75,0,0,.01)
phi4 = .9*sigmoid(X,Y,1,-1,75,0,0,.01)

theta = np.log(np.exp(phi1)+np.exp(phi2)+np.exp(phi3)+np.exp(phi4))
pattern = np.cos(theta)
fig, ax = plt.subplots()
im = ax.imshow(pattern)
plt.colorbar(im,ax=ax)
plt.show()
