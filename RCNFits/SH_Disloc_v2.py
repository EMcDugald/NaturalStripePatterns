import numpy as np
import scipy.io as sio
from scipy.fft import fft2, ifft2, fftfreq
import time
import os
import matplotlib.pyplot as plt

def solveSH(Lx,Ly,Nx,Ny,nwls,h,tmax,nsave,filename,Rscale=.5,amplitude=.1,energy=True, sig_scale=1.0):
    '''
    :param Lx: container length in x direction
    :param Ly: container length in y direction
    :param Nx: x discretization points
    :param Ny: y discretization points
    :param h: time step increment
    :param tmax: final time
    :param filename: string for saving data
    :param r: scales the R parameter in swift hohenberg
    :param beta: if solving on an ellipse, sets relative size of ellipse within Lx x Ly rectangle
    :param amplitude: sets amplitude for initial condition
    :param init_flag: determines a range of initial conditions: 1 is random on rectangle, 2 is
    sin function on rectangle, 3 is eikonal solution on ellipse
    :return: void: saves data
    '''
    print("starting method")
    xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
    yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
    X, Y = np.meshgrid(xx, yy)

    # set R function, if init_flag=3, we are on an ellipse
    R, u0 = make_ramp_init(Lx,Ly,Nx,Ny,Rscale,nwls,amplitude,sig_scale)

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

    tt = np.zeros(nsave+1)
    uu = np.zeros((Ny,Nx,nsave+1))
    ee = np.zeros((Ny,Nx,nsave+1))
    uu[:, :, 0] = u0
    ee[:, :, 0] = edensity(xi,eta,u0,ind,R)
    tt[0] = 0
    ii = 0
    start = time.time()

    nmax = int(np.round(tmax / h))
    idx_shift = int(np.floor(nmax / nsave))
    j = 0
    #begin time stepping
    for n in range(1, int(nmax) + 1):
        t = n * h
        print("step:",n)
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

        if ii % idx_shift == 0:
            uu[:, :,j+1] = u0
            tt[j+1] = t
            if energy:
                ee[:, :, j+1] = edensity(xi,eta,u0,ind,R)
            j += 1

        ii = ii+1


    end = time.time()
    print("time to generate solutions: ", end - start)
    if energy:
        mdict = {"tt": tt.reshape(1, len(tt)), "xx": xx.reshape(Nx, 1), "yy": yy.reshape(Ny, 1), "uu": uu,"ee": ee}
    else:
        mdict = {"tt": tt.reshape(1, len(tt)), "xx": xx.reshape(Nx, 1), "yy": yy.reshape(Ny, 1), "uu": uu}
    dir = os.getcwd()+"/data/sh_dislocation/"
    sio.savemat(dir+str(filename)+".mat", mdict)



# method to be called for setting initial condition for solution on ellipse

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

def sigmoid(x, sigscale):
    return 1 / (1 + np.exp(-sigscale*x))

def make_ramp_init(Lx,Ly,Nx,Ny,Rscale,nwls,amp,sigscale):
    xx = (Lx / Nx) * np.linspace(-Nx / 2 + 1, Nx / 2, Nx)
    yy = (Ly / Ny) * np.linspace(-Ny / 2 + 1, Ny / 2, Ny)
    X, Y = np.meshgrid(xx, yy)

    top = 2 * np.pi * sigmoid(X,sigscale) + 2 * np.pi * nwls
    bottom = -2 * np.pi * sigmoid(X,sigscale) - 2 * np.pi * nwls
    domain = np.where(((Y < top) & (Y > bottom))
                      &
                      ((X < 25 * np.pi) & (X > -25 * np.pi)),
                      1, 0)

    # smoothing the domain
    outer_indctr = np.where(domain == 0, 1, 0)
    inner_indctr = np.where(domain == 1, 1, 0)

    X_inner = X[np.where(inner_indctr == 1)]
    Y_inner = Y[np.where(inner_indctr == 1)]

    dist_from_domain = np.zeros((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            if outer_indctr[i, j] == 1:
                dist_from_domain[i, j] += np.min(np.sqrt((X[i, j] - X_inner) ** 2 + (Y[i, j] - Y_inner) ** 2))

    recip = 1. / (1. + (3 * dist_from_domain) ** 2)
    dom_with_decay = .5 * (domain + recip)

    from scipy.ndimage import gaussian_filter
    sigma = 2.0
    smoothed_domain = gaussian_filter(dom_with_decay, sigma=sigma)
    dom_bdry_x = X[np.where((.9999 <= smoothed_domain) & (smoothed_domain <= 1.0))]
    dom_bdry_y = Y[np.where((.9999 <= smoothed_domain) & (smoothed_domain <= 1.0))]
    R = 2*(np.tanh(10 * smoothed_domain) - .5)

    init_phase = np.zeros((Ny, Nx))
    for i in range(Ny):
        for j in range(Nx):
            if inner_indctr[i, j] == 1:
                init_phase[i, j] += np.min(np.sqrt((X[i, j] - dom_bdry_x) ** 2 + (Y[i, j] - dom_bdry_y) ** 2))


    init = np.sin(init_phase)
    return R*Rscale, amp*init


#solveSH(30*np.pi,30*np.pi,1024,1024,5,.5,8000,2,"SH_Disloc_v2_3",Rscale=.5,amplitude=.5,energy=True,sig_scale=1.0)
#solveSH(60*np.pi,30*np.pi,512,256,5,.5,8000,2,"SH_Disloc_v2_6",Rscale=.5,amplitude=.5,energy=True,sig_scale=.05)
#solveSH(30*np.pi,30*np.pi,512,512,5,.5,8000,2,"SH_Disloc_v2_6",Rscale=.5,amplitude=.5,energy=True,sig_scale=.05)
#solveSH(30*np.pi,30*np.pi,512,512,5,.5,8000,2,"SH_Disloc_v2_7",Rscale=.5,amplitude=.5,energy=True,sig_scale=.1)
solveSH(30*np.pi,30*np.pi,256,256,5,.5,4000,2,"SH_Disloc_v2_9",Rscale=.5,amplitude=.5,energy=True,sig_scale=.01)
#solveSH(30*np.pi,30*np.pi,512,512,5,.5,8000,2,"SH_Disloc_v2_8",Rscale=.9,amplitude=.5,energy=True,sig_scale=.2)
data = sio.loadmat(os.getcwd()+"/data/sh_dislocation/"+"SH_Disloc_v2_9.mat")
U = data['uu']
E = data['ee']
fig, ax = plt.subplots(nrows=2, ncols=1)
plt.subplots_adjust(wspace=.2,hspace=.2)
im0 = ax[0].imshow(U[:,:,-1],cmap='bwr')
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(E[:,:,-1],cmap='bwr')
plt.colorbar(im1,ax=ax[1])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/sh_dislocation/"+"SH_Disloc_v2_9.png")












