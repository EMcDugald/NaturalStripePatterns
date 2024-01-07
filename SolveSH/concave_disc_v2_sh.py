import numpy as np
import scipy.io as sio
from scipy.fft import fft2, ifft2, fftfreq
import time
import os
import matplotlib.pyplot as plt


def solveSH(Lx,Ly,Nx,Ny,h,tmax,save_indices,filename,amplitude=.9,angle_fctr=3.0,init_phase_scale=1.):
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
    #make the number of rows twice as big- we will be stacking the initial condition
    yytmp = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
    Xtmp, Ytmp = np.meshgrid(xx, yytmp)
    # set phase
    theta = (2. / angle_fctr) * (np.sqrt(Xtmp ** 2 + Ytmp ** 2)) ** (angle_fctr / 2.) * np.sin(
        (angle_fctr / 2.) * np.arctan2(Xtmp, Ytmp))

    Ly *= 2
    Ny *= 2
    yy = (Ly / Ny) * np.linspace(-Ny / 2 + 1, Ny / 2, Ny)

    R = np.ones(shape=(Ny,Nx))

    # set initial condition
    u0 = np.zeros(shape=(Ny, Nx))
    if angle_fctr >= 2.0:
        u0[0:int(Ny/2), :] = np.flip(np.cos(init_phase_scale*theta))
        u0[int(Ny/2):, :] = np.cos(init_phase_scale*theta)
    else:
        u0[0:int(Ny/2), :] = np.cos(init_phase_scale * theta)
        u0[int(Ny/2):, :] = np.flip(np.cos(init_phase_scale * theta))
    u0 *= amplitude

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
    #begin time stepping
    for n in range(1, int(nmax) + 1):
        print("step: ", n)
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

        if ii in save_indices:
            print("saving frame at:", ii)
            uu[:, :, ii] = u0
            tt[ii] = t
            ee[:, :, ii] = edensity(xi, eta, u0, ind, R)
            ii = ii + 1


    end = time.time()
    print("time to generate solutions: ", end - start)
    mdict = {"tt": tt.reshape(1, len(tt)), "xx": xx.reshape(Nx, 1), "yy": yy.reshape(Ny, 1), "uu": uu,"ee": ee}
    dir = os.getcwd()+"/data/"
    sio.savemat(dir+str(filename)+".mat", mdict)



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


nx = 512
ny = 512
nsave = 5
tmax = 100
dt = .5
save_indices = [j for j in range(int(tmax/dt)+1)]

angle_factor = 2.5 #1-2 gives pgb like structure, 2-3 gives concave disc of narrow to equal aspect, 3+ gives concave disc of wider aspect
init_phase_scale = .5 #controls how many stripes are in init. increasing this gives more stripes
figname = "SHCncvDisc_v2_alpha_25em1_ps_5em1_v2"

solveSH(60*np.pi,60*np.pi,nx,ny,dt,tmax,save_indices,figname,
        amplitude=.8,angle_fctr=angle_factor,init_phase_scale=init_phase_scale)

data = sio.loadmat(os.getcwd()+"/data/"+figname+".mat")
U = data['uu']
E = data['ee']
x = data['xx'].T[0,:]
y = data['yy'].T[0,:]
dx = x[1]-x[0]
dy = y[1]-y[0]

x_st = round(nx/4)
x_end = x_st+2*x_st
y_st = 0
y_end = 4*round(ny/4)

fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(13,42))

ax[0,0].imshow(U[:,:,0],cmap='gray')
ax[0,0].title.set_text('t={}: full field'.format(0*dt))
ax[0,1].imshow(U[y_st:y_end,x_st:x_end,0],cmap='gray')
ax[0,1].title.set_text('t={}: interier field'.format(0*dt))
ax[0,2].imshow(E[y_st:y_end,x_st:x_end,0],cmap='gray')
ax[0,2].title.set_text('t={}: interier energy'.format(0*dt))

ax[1,0].imshow(U[:,:,25],cmap='gray')
ax[1,0].title.set_text('t={}: full field'.format(25*dt))
ax[1,1].imshow(U[y_st:y_end,x_st:x_end,25],cmap='gray')
ax[1,1].title.set_text('t={}: interier field'.format(25*dt))
ax[1,2].imshow(E[y_st:y_end,x_st:x_end,25],cmap='gray')
ax[1,2].title.set_text('t={}: interier energy'.format(25*dt))

ax[2,0].imshow(U[:,:,50],cmap='gray')
ax[2,0].title.set_text('t={}: full field'.format(50*dt))
ax[2,1].imshow(U[y_st:y_end,x_st:x_end,50],cmap='gray')
ax[2,1].title.set_text('t={}: interier field'.format(50*dt))
ax[2,2].imshow(E[y_st:y_end,x_st:x_end,50],cmap='gray')
ax[2,2].title.set_text('t={}: interier energy'.format(50*dt))

ax[3,0].imshow(U[:,:,100],cmap='gray')
ax[3,0].title.set_text('t={}: full field'.format(100*dt))
ax[3,1].imshow(U[y_st:y_end,x_st:x_end,100],cmap='gray')
ax[3,1].title.set_text('t={}: interier field'.format(100*dt))
ax[3,2].imshow(E[y_st:y_end,x_st:x_end,100],cmap='gray')
ax[3,2].title.set_text('t={}: interier energy'.format(100*dt))

ax[4,0].imshow(U[:,:,200],cmap='gray')
ax[4,0].title.set_text('t={}: full field'.format(200*dt))
ax[4,1].imshow(U[y_st:y_end,x_st:x_end,200],cmap='gray')
ax[4,1].title.set_text('t={}: interier field'.format(200*dt))
ax[4,2].imshow(E[y_st:y_end,x_st:x_end,200],cmap='gray')
ax[4,2].title.set_text('t={}: interier energy'.format(200*dt))

print("total energy at time {}: ".format(0*dt), np.sum(E[:,:,0]*dx*dy))
print("total energy at time {}: ".format(25*dt), np.sum(E[:,:,25]*dx*dy))
print("total energy at time {}: ".format(50*dt), np.sum(E[:,:,50]*dx*dy))
print("total energy at time {}: ".format(100*dt), np.sum(E[:,:,100]*dx*dy))
print("total energy at time {}: ".format(200*dt), np.sum(E[:,:,200]*dx*dy))

for a in fig.axes:
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.set_aspect('equal')

fig.subplots_adjust(wspace=.2, hspace=.2)

plt.savefig(os.getcwd()+"/figs/"+figname+".png")












