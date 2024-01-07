import numpy as np
import matplotlib.pyplot as plt

# Lx = 60*np.pi
# Ly = 60*np.pi
# Nx = 256
# Ny = 256
# x = np.linspace(-Lx/2,Lx/2,Nx)
# y = np.linspace(-Ly/2,Ly/2,Ny)
# X,Y = np.meshgrid(x,y)
# pw0 = np.exp(1j*(1.*X+1.*Y))
# r_grid = np.sqrt(X**2+Y**2)
# theta_grid = np.arctan2(X,Y)
# alpha = np.pi/4
# Rscale = .9
# circ_mask = np.where((r_grid<Rscale*x[-1]),1,0)

# mask1 = np.where((-alpha/2. <= theta_grid) & (theta_grid < 0),1,0)
# mask2 = np.where((0. <= theta_grid) & (theta_grid < alpha/2.),1,0)
# mask3 = np.where((alpha/2. <= theta_grid) & (theta_grid < np.pi/2. + alpha/4.),1,0)
# mask4 = np.where((np.pi/2. + alpha/4. <= theta_grid) & (theta_grid < np.pi),1,0)
# mask5 = np.where((-np.pi <= theta_grid) & (theta_grid < -np.pi/2.-alpha/4.),1,0)
# mask6 = np.where((-np.pi/2.-alpha/4. <= theta_grid) & (theta_grid < -alpha/2.),1,0)

# fig = plt.figure()
# ax1 = fig.add_subplot(311)
# ax1.imshow((mask1 + mask2)*circ_mask)
# ax2 = fig.add_subplot(312)
# ax2.imshow((mask3 + mask4)*circ_mask)
# ax3 = fig.add_subplot(313)
# ax3.imshow((mask5 + mask6)*circ_mask)
# plt.show()

#no need to rotate this, since the symmetry angle is 0.
# kx1 = np.cos(alpha/2.)
# ky1 = np.sin(alpha/2.)
# kx2 = -np.cos(alpha/2.)
# ky2 = np.sin(alpha/2.)

# fig = plt.figure()
# pw1 = np.exp(1j*(kx1*X+ky1*Y))*mask1
# pw2 = np.exp(1j*(kx2*X+ky2*Y))*mask2
# ax1 = fig.add_subplot(111)
# ax1.imshow(np.real(pw1+pw2)*circ_mask)
# plt.show()

# symmetry angle is pi/2-alpha/4, and angle between pgbs is pi-alpha/2
# this is derived using the angle between the pgbs, and adding a rotation
# phi = np.pi/2. - alpha/4.
# kx3 = np.cos(np.pi/2.- alpha/4.)*np.cos(phi) - np.sin(np.pi/2.-alpha/4.)*np.sin(phi)
# ky3 = np.cos(np.pi/2 - alpha/4.)*np.sin(phi) + np.sin(np.pi/2.-alpha/4.)*np.cos(phi)
# kx4 = -np.cos(np.pi/2.- alpha/4.)*np.cos(phi) - np.sin(np.pi/2.-alpha/4.)*np.sin(phi)
# ky4 = -np.cos(np.pi/2 - alpha/4.)*np.sin(phi) + np.sin(np.pi/2.-alpha/4.)*np.cos(phi)

# fig = plt.figure()
# pw3 = np.exp(1j*(kx3*X+ky3*Y))*mask3
# pw4 = np.exp(1j*(kx4*X+ky4*Y))*mask4
# ax1 = fig.add_subplot(111)
# ax1.imshow(np.real(pw3+pw4)*circ_mask)
# plt.show()


# can use symmetry to make these
# kx5 = -kx4
# ky5 = ky4
# kx6 = -kx3
# ky6 = ky3

# fig = plt.figure()
# pw5 = np.exp(1j*(kx5*X+ky5*Y))*mask5
# pw6 = np.exp(1j*(kx6*X+ky6*Y))*mask6
# ax1 = fig.add_subplot(111)
# ax1.imshow(np.real(pw5+pw6)*circ_mask)
# plt.show()

# cd = np.real(pw1+pw2+pw3+pw4+pw5+pw6)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# im1 = ax1.imshow(cd*circ_mask)
# plt.colorbar(im1,ax=ax1)
# plt.show()

#from scipy.ndimage import gaussian_filter
# import scipy.ndimage as ndimage
# smoothed_cd = ndimage.uniform_filter(cd, size=10)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# im1 = ax1.imshow(smoothed_cd*circ_mask)
# plt.colorbar(im1,ax=ax1)
# plt.show()

def concave_disc(Lx,Ly,Nx,Ny,rad_scale,alpha,with_mask=True):
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(-Ly / 2, Ly / 2, Ny)
    X, Y = np.meshgrid(x, y)
    r_grid = np.sqrt(X ** 2 + Y ** 2)
    theta_grid = np.arctan2(X, Y)
    circ_mask = np.where((r_grid < rad_scale * x[-1]), 1, 0)
    mask1 = np.where((-alpha / 2. <= theta_grid) & (theta_grid < 0), 1, 0)
    mask2 = np.where((0. <= theta_grid) & (theta_grid < alpha / 2.), 1, 0)
    mask3 = np.where((alpha / 2. <= theta_grid) & (theta_grid < np.pi / 2. + alpha / 4.), 1, 0)
    mask4 = np.where((np.pi / 2. + alpha / 4. <= theta_grid) & (theta_grid < np.pi), 1, 0)
    mask5 = np.where((-np.pi <= theta_grid) & (theta_grid < -np.pi / 2. - alpha / 4.), 1, 0)
    mask6 = np.where((-np.pi / 2. - alpha / 4. <= theta_grid) & (theta_grid < -alpha / 2.), 1, 0)
    kx1 = np.cos(alpha / 2.)
    ky1 = np.sin(alpha / 2.)
    kx2 = -np.cos(alpha / 2.)
    ky2 = np.sin(alpha / 2.)
    phi = np.pi / 2. - alpha / 4.
    kx3 = np.cos(np.pi / 2. - alpha / 4.) * np.cos(phi) - np.sin(np.pi / 2. - alpha / 4.) * np.sin(phi)
    ky3 = np.cos(np.pi / 2 - alpha / 4.) * np.sin(phi) + np.sin(np.pi / 2. - alpha / 4.) * np.cos(phi)
    kx4 = -np.cos(np.pi / 2. - alpha / 4.) * np.cos(phi) - np.sin(np.pi / 2. - alpha / 4.) * np.sin(phi)
    ky4 = -np.cos(np.pi / 2 - alpha / 4.) * np.sin(phi) + np.sin(np.pi / 2. - alpha / 4.) * np.cos(phi)
    kx5 = -kx4
    ky5 = ky4
    kx6 = -kx3
    ky6 = ky3
    pw1 = np.exp(1j * (kx1 * X + ky1 * Y)) * mask1
    pw2 = np.exp(1j * (kx2 * X + ky2 * Y)) * mask2
    pw3 = np.exp(1j * (kx3 * X + ky3 * Y)) * mask3
    pw4 = np.exp(1j * (kx4 * X + ky4 * Y)) * mask4
    pw5 = np.exp(1j * (kx5 * X + ky5 * Y)) * mask5
    pw6 = np.exp(1j * (kx6 * X + ky6 * Y)) * mask6
    pat = pw1+pw2+pw3+pw4+pw5+pw6
    if with_mask:
        return np.real(pat)*circ_mask
    else:
        return np.real(pat)



Lx = 60 * np.pi
Ly = 60 * np.pi
Nx = 256
Ny = 256
alpha = np.pi/4.
Rscale = .9
cd1 = concave_disc(Lx,Ly,Nx,Ny,Rscale,alpha,with_mask=True)
cd1_no_mask = concave_disc(Lx,Ly,Nx,Ny,Rscale,alpha,with_mask=False)
alpha = np.pi/3.
cd2 = concave_disc(Lx,Ly,Nx,Ny,Rscale,alpha,with_mask=True)
cd2_no_mask = concave_disc(Lx,Ly,Nx,Ny,Rscale,alpha,with_mask=False)
alpha = 2*np.pi/3.
cd3 = concave_disc(Lx,Ly,Nx,Ny,Rscale,alpha,with_mask=True)
cd3_no_mask = concave_disc(Lx,Ly,Nx,Ny,Rscale,alpha,with_mask=False)
alpha = 2.5*np.pi/3.
cd4 = concave_disc(Lx,Ly,Nx,Ny,Rscale,alpha,with_mask=True)
cd4_no_mask = concave_disc(Lx,Ly,Nx,Ny,Rscale,alpha,with_mask=False)


fig = plt.figure(figsize=(9,17))
ax1 = fig.add_subplot(421)
ax1.imshow(cd1)
ax2 = fig.add_subplot(422)
ax2.imshow(cd1_no_mask)
ax3 = fig.add_subplot(423)
ax3.imshow(cd2)
ax4 = fig.add_subplot(424)
ax4.imshow(cd2_no_mask)
ax5 = fig.add_subplot(425)
ax5.imshow(cd3)
ax6 = fig.add_subplot(426)
ax6.imshow(cd3_no_mask)
ax7 = fig.add_subplot(427)
ax7.imshow(cd4)
ax8 = fig.add_subplot(428)
ax8.imshow(cd4_no_mask)


for a in fig.axes:
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.set_aspect('equal')

fig.subplots_adjust(wspace=0, hspace=0)

plt.show()