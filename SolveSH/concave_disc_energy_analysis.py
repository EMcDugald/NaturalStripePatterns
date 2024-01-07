import numpy as np
import scipy.io as sio
from scipy.fft import fft2, ifft2, fftfreq
import time
import os
import matplotlib.pyplot as plt
import sys

f = open(os.getcwd()+"/logs/concave_disclination_energy_analysis.out", 'w')
sys.stdout = f


def solveSH(Lx,Ly,Nx,Ny,alpha,h,tmax,save_indices,filename,Rscale=.5,rad_scale=.45,amplitude=.5):
    '''
    :param Lx: container length in x direction
    :param Ly: container length in y direction
    :param Nx: x discretization points
    :param Ny: y discretization points
    :param h: time step increment
    :param tmax: final time
    :param filename: string for saving data
    :param Rscale: scales the R parameter in swift hohenberg
    :param beta: if solving on an ellipse, sets relative size of ellipse within Lx x Ly rectangle
    :param amplitude: sets amplitude for initial condition
    :return: void: saves data
    '''
    xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
    yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
    X, Y = np.meshgrid(xx, yy)

    # set R function
    mask = np.tanh(np.sqrt(Lx**2+Ly**2)*(rad_scale-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)+1
    R = Rscale*(mask-1)

    # set initial condition
    init, mask1, mask2, mask3, mask4, mask5, mask6 = concave_disc(Lx,Ly,Nx,Ny,alpha)
    u0 = amplitude*init

    # -- precompute ETDRK4 scalar quantities --#
    kx = (2. * np.pi / Lx) * fftfreq(Nx, 1. / Nx)  # wave numbers
    ky = (2. * np.pi / Ly) * fftfreq(Ny, 1. / Ny)
    xi, eta = np.meshgrid(kx, ky)
    L = -(1 - xi ** 2 - eta ** 2) ** 2
    E = np.exp(h * L)
    E2 = np.exp(h * L / 2)

    M = 16  # number of points for complex means
    r = np.exp(1j * np.pi * ((np.arange(1, M + 1, 1) - .5) / M))  # roots of unity
    L2 = L.flatten()  # convert to single column
    LR = h * np.vstack([L2] * M).T + np.vstack([r] * Nx * Ny)  # adding r(j) to jth column
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, 1))  # means in the 2 directions
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, 1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, 1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, 1))

    f1 = np.reshape(f1, (Ny, Nx))
    f2 = np.reshape(f2, (Ny, Nx))
    f3 = np.reshape(f3, (Ny, Nx))
    Q = np.reshape(Q, (Ny, Nx))

    # dealiasing
    Fx = np.zeros((Nx, 1), dtype=bool)  # Fx = 1 for high frequencies which will be set to 0
    Fy = np.zeros((Ny, 1), dtype=bool)
    Fx[int(Nx / 2 - np.round(Nx / 4)):int(1 + Nx / 2 + np.round(Nx / 4))] = True
    Fy[int(Ny / 2 - np.round(Ny / 4)):int(1 + Ny / 2 + np.round(Ny / 4))] = True

    alxi, aleta = np.meshgrid(Fx, Fy)
    ind = alxi | aleta  # de-aliasing index

    # filter R an u0

    Rhat = fft2(R)
    Rhat[ind] = 0
    R = np.real(ifft2(Rhat))
    vv = fft2(u0)
    vv[ind] = 0
    u0 = np.real(ifft2(vv))
    Q[ind] = 0  # Q is the only term the multiplies the non linear factors

    tt = np.zeros(len(save_indices))
    uu = np.zeros((Ny, Nx, len(save_indices)))
    ee = np.zeros((Ny, Nx, len(save_indices)))
    start = time.time()
    nmax = int(np.round(tmax / h))

    if 0 in save_indices:
        tt[0] = 0.
        uu[:, :, 0] = u0
        ee[:, :, 0] = edensity(xi, eta, u0, ind, R)

    ii = 1
    for n in range(1, int(nmax) + 1):
       # print("on step:",n)
        t = n * h
        Nv = fft2(R * u0 - u0 ** 3)
        a = E2 * vv + Q * Nv
        ua = np.real(ifft2(a))
        Na = fft2(R * ua - ua ** 3)
        b = E2 * vv + Q * Na
        ub = np.real(ifft2(b))
        Nb = fft2(R * ub - ub ** 3)
        c = E2 * a + Q * (2 * Nb - Nv)
        uc = np.real(ifft2(c))
        Nc = fft2(R * uc - uc ** 3)
        vv = E * vv + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        u0 = np.real(ifft2(vv))

        if n in save_indices:
            uu[:, :, ii] = u0
            tt[ii] = t
            ee[:, :, ii] = edensity(xi, eta, u0, ind, R)
            ii = ii + 1

    end = time.time()
    print("time to generate solutions: ", end - start)
    mdict = {"tt": tt.reshape(1, len(tt)), "xx": xx.reshape(Nx, 1), "yy": yy.reshape(Ny, 1), "uu": uu, "ee": ee,
             "mask1": mask1,"mask2": mask2,"mask3": mask3,"mask4": mask4, "mask5": mask5, "mask6": mask6}
    dir = os.getcwd() + "/data/"
    sio.savemat(dir + str(filename) + ".mat", mdict)


def edensity(xi,eta,u0,ind,R):
    eloc = (1-xi**2-eta**2)*fft2(u0)
    eloc[ind] = 0
    eloc = np.real(ifft2(eloc)**2)

    u0sq = fft2(u0**2)
    u0sq[ind] = 0
    u0sq = np.real(ifft2(u0sq))

    u04th = fft2(u0sq**2)
    u04th[ind] = 0
    u04th = np.real(ifft2(u04th))
    return .5*(eloc-R*u0sq+.5*u04th)


def concave_disc(Lx,Ly,Nx,Ny,alpha):
    x = np.linspace(-Lx / 2, Lx / 2, Nx)
    y = np.linspace(-Ly / 2, Ly / 2, Ny)
    X, Y = np.meshgrid(x, y)
    theta_grid = np.arctan2(X, Y)
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
    return np.real(pat), mask1, mask2, mask3, mask4, mask5, mask6


nx = 512
ny = 512
nsave = 5
tmax = 250
dt = .5
Lx = 90*np.pi
Ly = 90*np.pi

save_indices = [0, 50, 499]

alpha1 = np.linspace(np.pi/16,2*np.pi/3,10)
alpha2 = np.linspace(2.3*np.pi/3,np.pi,5)
alphas = np.hstack((alpha1,alpha2))

for alpha in alphas:
    print("Generating SH Fields for alpha = {:.3e}".format(alpha))
    figname = "SHCncvDisc_v3_alpha_{:.3e}".format(alpha)

    solveSH(Lx,Ly,nx,ny,alpha,dt,tmax,save_indices,figname,
           Rscale=.5,rad_scale=.45,amplitude=.5)

    data = sio.loadmat(os.getcwd()+"/data/"+figname+".mat")
    U = data['uu']
    E = data['ee']
    x = data['xx'].T[0,:]
    y = data['yy'].T[0,:]
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    t = data['tt'][0]
    mask1 = data['mask1']
    mask2 = data['mask2']
    mask3 = data['mask3']
    mask4 = data['mask4']
    mask5 = data['mask5']
    mask6 = data['mask6']

    x_st = round(nx/4)
    x_end = x_st+2*x_st
    y_st = round(ny/4)
    y_end =y_st+2*y_st

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(13,15))

    ax[0,0].imshow(U[:,:,0],cmap='gray')
    ax[0,0].title.set_text('t={:.3e}: full field'.format(t[0]))
    ax[0,1].imshow(U[y_st:y_end,x_st:x_end,0],cmap='gray')
    ax[0,1].title.set_text('t={:.3e}: interier field'.format(t[0]))
    ax[0,2].imshow(E[y_st:y_end,x_st:x_end,0],cmap='gray')
    ax[0,2].title.set_text('t={:.3e}: interier energy'.format(t[0]))

    ax[1,0].imshow(U[:,:,1],cmap='gray')
    ax[1,0].title.set_text('t={:.3e}: full field'.format(t[1]))
    ax[1,1].imshow(U[y_st:y_end,x_st:x_end,1],cmap='gray')
    ax[1,1].title.set_text('t={:.3e}: interier field'.format(t[1]))
    ax[1,2].imshow(E[y_st:y_end,x_st:x_end,1],cmap='gray')
    ax[1,2].title.set_text('t={:.3e}: interier energy'.format(t[1]))

    ax[2,0].imshow(U[:,:,2],cmap='gray')
    ax[2,0].title.set_text('t={:.3e}: full field'.format(t[2]))
    ax[2,1].imshow(U[y_st:y_end,x_st:x_end,2],cmap='gray')
    ax[2,1].title.set_text('t={:.3e}: interier field'.format(t[2]))
    ax[2,2].imshow(E[y_st:y_end,x_st:x_end,2],cmap='gray')
    ax[2,2].title.set_text('t={:.3e}: interier energy'.format(t[2]))

    print("total energy at time {:.3e}: ".format(t[0]), np.sum(E[:,:,0]*dx*dy))
    print("total energy at time {:.3e}: ".format(t[1]), np.sum(E[:,:,1]*dx*dy))
    print("total energy at time {:.3e}: ".format(t[2]), np.sum(E[:,:,2]*dx*dy))
    print("energy of lower third {:.3e}: ".format(t[2]), np.sum(E[:, :, 2] * dx * dy * (mask1 + mask2)))
    print("energy of right third {:.3e}: ".format(t[2]), np.sum(E[:, :, 2] * dx * dy * (mask3 + mask4)))
    print("energy of left third {:.3e}: ".format(t[2]), np.sum(E[:, :, 2] * dx * dy * (mask5 + mask6)))


    for a in fig.axes:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')

    fig.subplots_adjust(wspace=.2, hspace=.2)
    plt.savefig(os.getcwd()+"/figs/energy_analysis/"+figname+".png")
    plt.close()

    fig, ax = plt.subplots(nrows=2,ncols=2)
    ax[0,0].plot(U[:,int(nx/2),2])
    ax[0,0].title.set_text("Field cross section along x=0")
    ax[1,0].plot(E[:,int(nx/2), 2])
    ax[1,0].title.set_text("Energy cross section along x=0")
    ax[0, 1].plot(U[int(ny/2),:, 2])
    ax[0, 1].title.set_text("Field cross section along x=0")
    ax[1, 1].plot(E[int(ny/2),:, 2])
    ax[1, 1].title.set_text("Energy cross section along x=0")
    fig.subplots_adjust(wspace=.2, hspace=.2)
    plt.savefig(os.getcwd() + "/figs/energy_analysis/" + figname + "_profiles"+".png")
    plt.close()

f.close()








