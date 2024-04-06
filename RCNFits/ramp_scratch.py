import numpy as np
import matplotlib.pyplot as plt
import os

def sigmoid(x):
    return 1/(1+np.exp(-x))

Lx = 60*np.pi
Ly = 30*np.pi
Nx = 256
Ny = 128
xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
X, Y = np.meshgrid(xx, yy)

top = 2*np.pi*sigmoid(X)+2*np.pi*5
bottom = -2*np.pi*sigmoid(X) - 2*np.pi*5
domain = np.where(((Y < top) & (Y > bottom))
                 &
                 ((X<25*np.pi) & (X>-25*np.pi)),
                 1,0)

fig1, ax1 = plt.subplots(figsize=(8,4))
dom = ax1.imshow(domain,cmap='bwr')
fig1.colorbar(dom,ax=ax1)
plt.show()


#smoothing the domain
outer_indctr = np.where(domain==0,1,0)
inner_indctr = np.where(domain==1,1,0)

X_outer = X[np.where(outer_indctr==1)]
Y_outer = Y[np.where(outer_indctr==1)]
X_inner = X[np.where(inner_indctr==1)]
Y_inner = Y[np.where(inner_indctr==1)]

dist_from_domain = np.zeros((Ny,Nx))
for i in range(Ny):
    for j in range(Nx):
        if outer_indctr[i,j] == 1:
            dist_from_domain[i,j] += np.min(np.sqrt((X[i,j]-X_inner)**2+(Y[i,j]-Y_inner)**2))

fig2, ax2 = plt.subplots(figsize=(8,4))
dist_bdry = ax2.imshow(dist_from_domain,cmap='bwr')
fig2.colorbar(dist_bdry,ax=ax2)
plt.show()


recip = 1./(1. + (3*dist_from_domain)**2)
dom_with_decay = .5*(domain + recip)
fig3, ax3 = plt.subplots(figsize=(8,4))
dom_decay = ax3.imshow(dom_with_decay,cmap='bwr')
fig3.colorbar(dom_decay,ax=ax3)
plt.show()


from scipy.ndimage import gaussian_filter
sigma = 2.0
smoothed_domain = gaussian_filter(dom_with_decay, sigma=sigma)

fig4, ax4 = plt.subplots(figsize=(8,4))
smooth_dom = ax4.imshow(smoothed_domain,cmap='bwr')
fig4.colorbar(smooth_dom,ax=ax4)
plt.show()

dom_bdry_x = X[np.where((.9999<=smoothed_domain) & (smoothed_domain <= 1.0))]
dom_bdry_y = Y[np.where((.9999<=smoothed_domain) & (smoothed_domain <= 1.0))]
print(np.shape(dom_bdry_x))
fig5, ax5 = plt.subplots(figsize=(8,4))
dom_bdry = ax5.scatter(dom_bdry_x,dom_bdry_y,c='r')
plt.show()

R = 2*(np.tanh(10*smoothed_domain)-.5)
print(np.max(R))
print(np.min(R))
fig6, ax6 = plt.subplots(figsize=(8,4))
Rim = ax6.imshow(R,cmap='bwr')
fig6.colorbar(Rim,ax=ax6)
plt.show()

init_phase = np.zeros((Ny,Nx))
for i in range(Ny):
    for j in range(Nx):
        if inner_indctr[i,j] == 1:
            init_phase[i,j] += np.min(np.sqrt((X[i,j]-dom_bdry_x)**2+(Y[i,j]-dom_bdry_y)**2))

fig7, ax7 = plt.subplots(figsize=(8,4))
ip = ax7.imshow(init_phase,cmap='bwr')
fig7.colorbar(ip,ax=ax7)
plt.show()

init = np.sin(init_phase)
fig8, ax8 = plt.subplots(figsize=(8,4))
ipat = ax8.imshow(init,cmap='bwr')
fig8.colorbar(ipat,ax=ax8)
plt.show()







####################

