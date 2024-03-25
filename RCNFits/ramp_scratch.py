import numpy as np
import matplotlib.pyplot as plt
import os

Lx = 20
Ly = 10
Nx = 256
Ny = 128
xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
X, Y = np.meshgrid(xx, yy)

# Make a circle
cx = -Lx/8.
cy = 0.
r = Ly/3.
circle = np.where((X-cx)**2+(Y-cy)**2 < r**2,1,0)

# Make a trapezoid that intersects the circle
trap1_overlap = Lx/20. #distance that trapezoid overlaps circle
trap1_left_x = cx + r - trap1_overlap #x coordinate of left leg of trapezoid
trap1_top_left_y = np.sqrt(r**2-(trap1_left_x-cx)**2) #y coordinate of top left corner of trap
trap1_bottom_left_y = -np.sqrt(r**2-(trap1_left_x-cx)**2) #y coordinate of lower left corner of trap
trap1_left_side_length = trap1_top_left_y - trap1_bottom_left_y #length of left side of trap
trap1_right_side_length = .7*trap1_left_side_length #length of right side of trap
trap1_height = .7*r #distance between left and right leg of trap
trap1_top_right_y = trap1_right_side_length/2. #y coordinate of top right corner of trap
trap1_bottom_right_y = -trap1_right_side_length/2. #y coordinate of lower right corner of trap
trap1_right_x = trap1_left_x+trap1_height #x coordinate of right leg of trap
trap1_top_slope = (trap1_top_right_y-trap1_top_left_y)/(trap1_right_x-trap1_left_x) #slope of top leg of trap
trap1_bottom_slope = (trap1_bottom_right_y-trap1_bottom_left_y)/(trap1_right_x-trap1_left_x) # slope of bottom leg of trap
trap1_top_b = trap1_top_left_y - trap1_top_slope*trap1_left_x #y intercept of line defined by top leg
trap1_bottom_b = trap1_bottom_left_y - trap1_bottom_slope*trap1_left_x #y intercept of line defined by bottom leg
trap1 = np.where(((trap1_left_x < X) & (X < trap1_right_x))
                 &
                 ((trap1_bottom_slope*X + trap1_bottom_b < Y) & (Y < trap1_top_slope*X + trap1_top_b)),
                 1,0)

#make another trapezoid that touches the first trapezoid
trap2_left_x = trap1_right_x
trap2_top_left_y = trap1_top_right_y
trap2_bottom_left_y = trap1_bottom_right_y
trap2_left_side_length = trap2_top_left_y-trap2_bottom_left_y
trap2_height = trap1_height
trap2_right_x = trap2_left_x+trap2_height
trap2_right_side_length = 2*trap2_left_side_length
trap2_top_right_y = trap2_right_side_length/2.
trap2_bottom_right_y = -trap2_right_side_length/2.
trap2_top_slope = (trap2_top_right_y-trap2_top_left_y)/(trap2_right_x-trap2_left_x)
trap2_bottom_slope = (trap2_bottom_right_y-trap2_bottom_left_y)/(trap2_right_x-trap2_left_x)
trap2_top_b = trap2_top_left_y - trap2_top_slope*trap2_left_x #y intercept of line defined by top leg
trap2_bottom_b = trap2_bottom_left_y - trap2_bottom_slope*trap2_left_x
trap2 = np.where(((trap2_left_x < X) & (X < trap2_right_x))
                 &
                 ((trap2_bottom_slope*X + trap2_bottom_b < Y) & (Y < trap2_top_slope*X + trap2_top_b)),
                 1,0)

rect_left_x = trap2_right_x
rect_width = trap2_height/2.
rect_right_x = rect_left_x + rect_width
rect_top_y = trap2_top_right_y
rect_bottom_y = trap2_bottom_right_y
rect = np.where(((rect_left_x < X) & (X < rect_right_x))
                 &
                 ((rect_bottom_y < Y) & (Y < rect_top_y)),
                 1,0)

domain = circle+trap1+trap2+rect
domain = np.where(domain >= 1,1,0)

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

recip = 1./(1. + (1.5*dist_from_domain)**2)
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

init_phase1 = np.zeros((Ny,Nx))
for i in range(Ny):
    for j in range(Nx):
        if inner_indctr[i,j] == 1:
            init_phase1[i,j] += np.min(np.sqrt((X[i,j]-dom_bdry_x)**2+(Y[i,j]-dom_bdry_y)**2))

init_phase1 = 10*init_phase1
fig6, ax6 = plt.subplots(figsize=(8,4))
ip1 = ax6.imshow(init_phase1,cmap='bwr')
fig6.colorbar(ip1,ax=ax6)
plt.show()


pattern1 = np.sin(init_phase1)
fig7, ax7 = plt.subplots(figsize=(8,4))
p1 = ax7.imshow(pattern1,cmap='bwr')
fig7.colorbar(p1,ax=ax7)
plt.show()


R = np.tanh(10*smoothed_domain)-.5
print(np.max(R))
print(np.min(R))
fig8, ax8 = plt.subplots(figsize=(8,4))
Rim = ax8.imshow(R,cmap='bwr')
fig8.colorbar(Rim,ax=ax8)
plt.show()

from scipy import signal
est_phase = np.zeros(shape=(Ny,Nx))
unwrapped_est_phase = np.zeros(shape=(Ny,Nx))
print("debug")
for i in np.where(X[0,:]<0)[0]:
    if np.any(inner_indctr[:,i]==1):
        profile = pattern1[np.where(np.logical_and(inner_indctr[:,i]==1, Y[:,i]<0)==True),i]
        hilbert = signal.hilbert(profile)
        est_phase[np.where(np.logical_and(inner_indctr[:,i]==1, Y[:,i]<0)==True),i] += np.arctan2(np.imag(hilbert),np.real(hilbert))
        unwrapped_est_phase[np.where(np.logical_and(inner_indctr[:,i]==1, Y[:,i]<0)==True),i] += np.unwrap(est_phase[np.where(np.logical_and(inner_indctr[:,i]==1, Y[:,i]<0)==True),i])

fig9, ax9 = plt.subplots(figsize=(8,4))
ep_im = ax9.imshow(est_phase,cmap='bwr')
fig9.colorbar(ep_im,ax=ax9)
plt.show()

fig10, ax10 = plt.subplots(figsize=(8,4))
uep_im = ax10.imshow(unwrapped_est_phase,cmap='bwr')
fig10.colorbar(uep_im,ax=ax10)
plt.show()

smooth_unwrapped_phase = np.zeros(shape=(Ny,Nx))
for i in np.where(Y[:,0]<0)[0]:
    if np.any(inner_indctr[:, i] == 1):
        smooth_unwrapped_phase[i,np.where(np.logical_and(inner_indctr[i,:]==1, X[i,:]<0)==True)] = np.unwrap(unwrapped_est_phase[i,np.where(np.logical_and(inner_indctr[i,:]==1, X[i,:]<0)==True)])

fig11, ax11 = plt.subplots(figsize=(8,4))
suep_im = ax11.imshow(smooth_unwrapped_phase,cmap='bwr')
fig11.colorbar(suep_im,ax=ax11)
plt.show()
