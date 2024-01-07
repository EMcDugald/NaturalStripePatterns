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
angle_fctr = 1.25
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

##################################################


### SET 2 ###
angle_fctr = 1.5
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

###############

### SET 3 ###
angle_fctr = 1.75
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
angle_fctr = 2
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


### SET 5 ###
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


### SET 6 ###
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

### SET 7 ###
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


### SET 8 ###
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

### PERIODICITY TEST ###
pattern = np.zeros(shape=(2*Ny,Nx))
pattern[0:Ny,:] = np.flip(np.cos(theta))
pattern[Ny:,:] = np.cos(theta)
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,8))
im1 = ax.imshow(pattern)
plt.show()



### SET 9 ###
angle_fctr = 1.75
theta = (2./angle_fctr)*(np.sqrt(X**2+Y**2))**(angle_fctr/2.)*np.sin((angle_fctr/2.)*np.arctan2(X,Y))
R = Rscale*(np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1)

### PERIODICITY TEST 2 ###
pattern = np.zeros(shape=(2*Ny,Nx))
pattern[0:Ny,:] = np.cos(theta)
pattern[Ny:,:] = np.flip(np.cos(theta))
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(5,8))
im1 = ax.imshow(pattern)
plt.show()


# ### POLAR COORD STUFF ###
# angle_fctr = 3 #default is 3 for equal angles
# r = np.linspace(0,np.sqrt(50*np.pi),128)
# alpha = np.linspace(0,2*np.pi,256)
# R, Alpha = np.meshgrid(r,alpha)
# theta = (2./angle_fctr)*(R**(angle_fctr/2.))*np.sin((angle_fctr/2.)*Alpha)
# X = R*np.cos(Alpha)
# Y = R*np.sin(Alpha)
# fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(5,8))
# im1 = ax[0].scatter(X,Y,c=theta,cmap='bwr',alpha=.1)
# plt.colorbar(im1,ax=ax[0])
# im2 = ax[1].scatter(X,Y,c=np.cos(theta),cmap='bwr',alpha=.1)
# plt.show()

######### OLD ############

# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.linspace(-10,10,257)
# y = np.linspace(-10,10,257)
# X,Y = np.meshgrid(x,y)
#
# N=257
# y_set = np.zeros((N,N))
# for i in range(len(y_set)):
#     if i <= 128:
#         lower = -np.pi + (-10*np.pi/128)*i
#         upper = np.pi + (10*np.pi/128)*i
#     else:
#         lower = -11*np.pi + (10*np.pi/128)*(i-128)
#         upper = 11*np.pi + (-10*np.pi/128)*(i-128)
#     y_set[:, i] = np.linspace(lower, upper, 257)
#
# fig, ax = plt.subplots(nrows=2, ncols=1)
# ax[0].imshow(y_set)
# ax[1].imshow(np.cos(y_set))
# plt.show()
#
# N=257
# y_set2 = np.zeros((N,N))
# for i in range(len(y_set2)):
#     if i <= 128:
#         lower = -np.pi + (-10*np.pi/128)*i
#         upper = np.pi + (10*np.pi/128)*i
#     else:
#         lower = -11*np.pi + (10*np.pi/128)*(i-128)
#         upper = 11*np.pi + (-10*np.pi/128)*(i-128)
#     y_set2[:, i] = np.hstack(
#         (np.linspace(0, upper, 129),
#         np.linspace(lower,0,128))
#     )
#
#
# fig, ax = plt.subplots(nrows=2, ncols=1)
# ax[0].imshow(y_set2)
# ax[1].imshow(np.cos(y_set2))
# plt.show()