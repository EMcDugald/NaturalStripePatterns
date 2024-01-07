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


### PERIODICITY TEST 0 ###
angle_fctr = 1.1
theta = (2./angle_fctr)*(np.sqrt(X**2+Y**2))**(angle_fctr/2.)*np.sin((angle_fctr/2.)*np.arctan2(X,Y))
#R = Rscale*(np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1)

pattern = np.zeros(shape=(2*Ny,Nx))
pattern[0:Ny,:] = np.cos(theta)
pattern[Ny:,:] = np.flip(np.cos(theta))
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,8))
im1 = ax.imshow(pattern)
plt.show()

### PERIODICITY TEST 1 ###
angle_fctr = 1.5
theta = (2./angle_fctr)*(np.sqrt(X**2+Y**2))**(angle_fctr/2.)*np.sin((angle_fctr/2.)*np.arctan2(X,Y))
#R = Rscale*(np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1)

pattern = np.zeros(shape=(2*Ny,Nx))
pattern[0:Ny,:] = np.cos(theta)
pattern[Ny:,:] = np.flip(np.cos(theta))
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,8))
im1 = ax.imshow(pattern)
plt.show()

### PERIODICITY TEST 2 ###
angle_fctr = 1.9
theta = (2./angle_fctr)*(np.sqrt(X**2+Y**2))**(angle_fctr/2.)*np.sin((angle_fctr/2.)*np.arctan2(X,Y))
#R = Rscale*(np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1)

pattern = np.zeros(shape=(2*Ny,Nx))
pattern[0:Ny,:] = np.cos(theta)
pattern[Ny:,:] = np.flip(np.cos(theta))
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,8))
im1 = ax.imshow(pattern)
plt.show()


### PERIODICITY TEST 3 ###
angle_fctr = 2.0
theta = (2./angle_fctr)*(np.sqrt(X**2+Y**2))**(angle_fctr/2.)*np.sin((angle_fctr/2.)*np.arctan2(X,Y))
#R = Rscale*(np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1)

pattern = np.zeros(shape=(2*Ny,Nx))
pattern[0:Ny,:] = np.cos(theta)
pattern[Ny:,:] = np.flip(np.cos(theta))
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,8))
im1 = ax.imshow(pattern)
plt.show()


### PERIODICITY TEST 4 ###
angle_fctr = 2.5
theta = (2./angle_fctr)*(np.sqrt(X**2+Y**2))**(angle_fctr/2.)*np.sin((angle_fctr/2.)*np.arctan2(X,Y))
#R = Rscale*(np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1)

pattern = np.zeros(shape=(2*Ny,Nx))
pattern[0:Ny,:] = np.flip(np.cos(theta))
pattern[Ny:,:] = np.cos(theta)
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,8))
im1 = ax.imshow(pattern)
plt.show()


### PERIODICITY TEST 5 ###
angle_fctr = 3.0
theta = (2./angle_fctr)*(np.sqrt(X**2+Y**2))**(angle_fctr/2.)*np.sin((angle_fctr/2.)*np.arctan2(X,Y))
#R = Rscale*(np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1)

pattern = np.zeros(shape=(2*Ny,Nx))
pattern[0:Ny,:] = np.flip(np.cos(theta))
pattern[Ny:,:] = np.cos(theta)
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,8))
im1 = ax.imshow(pattern)
plt.show()