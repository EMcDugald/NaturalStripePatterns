import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, fftshift
import os
import matplotlib.pyplot as plt
import sys
from scipy import signal
import time

f = open(os.getcwd()+"/logs/spectrum_analysis.out", 'w')
sys.stdout = f

def solveSH(Lx,Ly,Nx,Ny,h,tmax,Rscale=.5,beta=.45,amplitude=.1,init_flag=1):
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
    xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
    yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
    X, Y = np.meshgrid(xx, yy)

    # set R function, if init_flag=3, we are on an ellipse
    if init_flag==3:
        R = Rscale*np.tanh(np.sqrt(Lx**2+Ly**2)*(beta-np.sqrt((X/Lx)**2+(Y/Ly)**2))/2)
    else:
        R = Rscale*np.ones((Ny, Nx))

    # set initial condition, init_flag=3 means we are on the ellipse
    if init_flag == 1:
        u0 = np.random.randn(Ny, Nx)
        u0 = amplitude * u0 / np.linalg.norm(u0, np.inf)
    elif init_flag == 2:
        u0 = amplitude * np.sin(Y)
    else:
        u0 = ellipse_init(X, Y, beta * Lx, beta * Ly, amplitude)

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

    num_steps = round(tmax/h)
    tt = np.zeros(num_steps+1)
    uu = np.zeros((Ny,Nx,num_steps+1))
    ee = np.zeros((Ny,Nx,num_steps+1))
    uu[:, :, 0] = u0
    ee[:, :, 0] = edensity(xi,eta,u0,ind,R)
    tt[0] = 0
    start = time.time()
    for n in range(1, num_steps + 1):
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
        uu[:, :,n] = u0
        tt[n] = t
        ee[:, :, n] = edensity(xi,eta,u0,ind,R)
    end = time.time()
    print("time to generate solutions: ", end - start)
    return uu, ee, X, Y, tt, xi, eta



# method to be called for setting initial condition for solution on ellipse
def ellipse_init(X,Y,a,b,amp):
    nmx = 256
    q = 2*np.pi*np.arange(1,nmx+1,1)/nmx
    imx, jmx = np.shape(X)
    bdry = np.vstack((a*np.cos(q), b*np.sin(q)))
    rho = np.zeros((imx,jmx))
    for ii in range(imx):
        for jj in range(jmx):
            rho[ii,jj] = np.min((X[ii,jj]-bdry[0,:])**2+(Y[ii,jj]-bdry[1,:])**2)
    kx = (np.pi/a)*fftfreq(jmx,1./jmx)
    ky = (np.pi/b)*fftfreq(imx,1./imx)
    xi, eta = np.meshgrid(kx, ky)
    rho = ifft2(np.exp(-(xi**2+eta**2))*fft2(rho))
    return np.real(amp*np.sin(np.sqrt(rho)))

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


Nx=128
Ny=128
U, E, X, Y, t, Kx, Ky = solveSH(20*np.pi,20*np.pi,Nx,Ny,.5,500,Rscale=.5,beta=.45,amplitude=.1,init_flag=1)

fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(6,14))
im1 = ax[0].scatter(X.flatten(),Y.flatten(),c=U[:,:,-1].flatten(),cmap='bwr')
im2 = ax[1].scatter(Kx.flatten(),Ky.flatten(),c=np.real(fft2(U[:,:,-1])).flatten(),cmap='bwr')
im3 = ax[2].scatter(Kx.flatten(),Ky.flatten(),c=np.imag(fft2(U[:,:,-1])).flatten(),cmap='bwr')
ax[0].title.set_text('Field')
ax[1].title.set_text('Real Spectrum')
ax[2].title.set_text('Imaginary Spectrum')
ax[0].set_xlim(X[0,0],X[0,-1])
ax[0].set_ylim(Y[0,0],Y[-1,0])
ax[1].set_xlim(fftshift(Kx)[0,0],fftshift(Kx)[0,-1])
ax[1].set_ylim(fftshift(Ky)[0,0],fftshift(Ky)[-1,0])
ax[2].set_xlim(fftshift(Kx)[0,0],fftshift(Kx)[0,-1])
ax[2].set_ylim(fftshift(Ky)[0,0],fftshift(Ky)[-1,0])
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])
plt.tight_layout()
plt.savefig(os.getcwd()+"/Spectrum1_scatter.png")
plt.close()

fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(6,14))
im1 = ax[0].imshow(U[:,:,-1],cmap='bwr')
im2 = ax[1].imshow(fftshift(np.real(fft2(U[:,:,-1]))),cmap='bwr')
im3 = ax[2].imshow(fftshift(np.imag(fft2(U[:,:,-1]))),cmap='bwr')
ax[0].title.set_text('Field')
ax[1].title.set_text('Real Spectrum')
ax[2].title.set_text('Imaginary Spectrum')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])
plt.tight_layout()
plt.savefig(os.getcwd()+"/Spectrum1_imshow.png")
plt.close()


spec = fftshift(fft2(U[:,:,-1]))
indices = np.argsort(-np.abs(spec).flatten())
Kxs = fftshift(Kx).flatten()[indices]
Kys = fftshift(Ky).flatten()[indices]
FourierCoeffs = spec.flatten()[indices]

plane_waves = []
exp_sum = np.zeros(shape=np.shape(spec),dtype='complex')
i = 0
for fourier_coeff in FourierCoeffs:
    plane_wave = (fourier_coeff*np.exp(1j*(Kxs[i]*X+Kys[i]*Y)))/(Nx*Ny)
    exp_sum += plane_wave
    print("on step: ", i)
    print("FC: ",fourier_coeff, "Kx:", Kxs[i], "Ky: ", Kys[i])
    err = np.linalg.norm(U[:,:,-1]-np.roll(fftshift(np.real(exp_sum)),axis=(0,1),shift=(1,1)))
    rel_err = err/np.linalg.norm(U[:,:,-1])
    print("Err:", err, "relErr:", rel_err)
    plane_waves.append(np.roll(fftshift(np.real(np.exp(1j*(Kxs[i]*X+Kys[i]*Y)))), axis=(0, 1), shift=(1, 1)))
    if rel_err < 5e-2:
        print("number of sum terms:", i)
        fig, ax = plt.subplots(nrows=2, ncols=1)
        im1 = ax[0].imshow(U[:,:,-1], cmap='bwr')
        plt.colorbar(im1, ax=ax[0])
        ax[0].title.set_text('SH Field')
        im2 = ax[1].imshow(np.roll(fftshift(np.real(exp_sum)),axis=(0,1),shift=(1,1)), cmap='bwr')
        plt.colorbar(im2, ax=ax[1])
        ax[1].title.set_text('Reconstruct from {} FCs'.format(i))
        plt.tight_layout()
        plt.savefig(os.getcwd() + "/Spectrum1_FourierRecon.png")
        plt.close()
        break
    i += 1

print("debug")
# min_num_periods = min(Lx,Ly)/(2*np.pi)
# max_num_periods = np.sqrt(Lx**2+Ly**2)/(2*np.pi)
#ToDo: determine window size of plane wave using wave vec diraction and dimensions of domain
est_kx = np.zeros(shape=np.shape(spec))
est_ky = np.zeros(shape=np.shape(spec))
corr_0 = np.zeros(shape=np.shape(spec))
midx = int(Nx/2)
midy = int(Ny/2)
num_pts = 8
print("Total coeffs:", i)
for k in range(i):
    pw = np.real((FourierCoeffs[k]*np.exp(1j*(Kxs[k]*X+Kys[k]*Y)))/(Nx*Ny))
    corr_next = signal.correlate2d(U[:,:,-1],
                                   fftshift(pw[int(Ny/2)-num_pts:int(Ny/2)+num_pts,int(Ny/2)-num_pts:int(Ny/2)+num_pts]),
                                   boundary='wrap', mode='same')
    est_kx[np.where(corr_next>corr_0)] = Kxs[k]
    est_ky[np.where(corr_next>corr_0)] = Kys[k]
    if k % 100 == 0:
        print("kx = ", Kxs[k], "Ky = ",Kys[k])
    corr_0 = corr_next

fig, ax = plt.subplots(nrows=2, ncols=2)
im1 = ax[0,0].imshow(est_kx, cmap='bwr')
plt.colorbar(im1, ax=ax[0,0])
ax[0,0].title.set_text('est_kx')
im2 = ax[0,1].imshow(est_ky, cmap='bwr')
plt.colorbar(im2, ax=ax[0,1])
ax[0,1].title.set_text('est_ky')
im3 = ax[1,0].imshow(np.sqrt(est_kx**2+est_ky**2), cmap='bwr')
plt.colorbar(im3, ax=ax[1,0])
ax[1,0].title.set_text('est wave num')
im4 = ax[1,1].imshow(U[:,:,-1], cmap='bwr')
plt.colorbar(im4, ax=ax[1,1])
ax[1,1].title.set_text('Field')
plt.tight_layout()
plt.savefig(os.getcwd() + "/Spectrum1_EstWaveVec.png")
plt.close()


###############################################################################################################

Nx = 256
Ny = 128
U, E, X, Y, t, Kx, Ky = solveSH(80*np.pi,40*np.pi,Nx,Ny,.5,100,Rscale=.5,beta=.45,amplitude=.1,init_flag=3)

fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(12,14))
im1 = ax[0].scatter(X.flatten(),Y.flatten(),c=U[:,:,-1].flatten(),cmap='bwr')
im2 = ax[1].scatter(Kx.flatten(),Ky.flatten(),c=np.real(fft2(U[:,:,-1])).flatten(),cmap='bwr')
im3 = ax[2].scatter(Kx.flatten(),Ky.flatten(),c=np.imag(fft2(U[:,:,-1])).flatten(),cmap='bwr')
ax[0].title.set_text('Field')
ax[1].title.set_text('Real Spectrum')
ax[2].title.set_text('Imaginary Spectrum')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])
plt.tight_layout()
plt.savefig(os.getcwd()+"/Spectrum2_scatter.png")
plt.close()

fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(12,14))
im1 = ax[0].imshow(U[:,:,-1],cmap='bwr')
im2 = ax[1].imshow(fftshift(np.real(fft2(U[:,:,-1]))),cmap='bwr')
im3 = ax[2].imshow(fftshift(np.imag(fft2(U[:,:,-1]))),cmap='bwr')
ax[0].title.set_text('Field')
ax[1].title.set_text('Real Spectrum')
ax[2].title.set_text('Imaginary Spectrum')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])
plt.tight_layout()
plt.savefig(os.getcwd()+"/Spectrum2_imshow.png")
plt.close()

spec = fftshift(fft2(U[:,:,-1]))
indices = np.argsort(-np.abs(spec).flatten())
Kxs = fftshift(Kx).flatten()[indices]
Kys = fftshift(Ky).flatten()[indices]
FourierCoeffs = spec.flatten()[indices]

plane_waves = []
exp_sum = np.zeros(shape=np.shape(spec),dtype='complex')
i = 0
for fourier_coeff in FourierCoeffs:
    plane_wave = (fourier_coeff*np.exp(1j*(Kxs[i]*X+Kys[i]*Y)))/(Nx*Ny)
    exp_sum += plane_wave
    print("on step: ", i)
    print("FC: ",fourier_coeff, "Kx:", Kxs[i], "Ky: ", Kys[i])
    err = np.linalg.norm(U[:,:,-1]-np.roll(fftshift(np.real(exp_sum)),axis=(0,1),shift=(1,1)))
    rel_err = err/np.linalg.norm(U[:,:,-1])
    print("Err:", err, "relErr:", rel_err)
    plane_waves.append(np.roll(fftshift(np.real(np.exp(1j*(Kxs[i]*X+Kys[i]*Y)))),axis=(0,1),shift=(1,1)))
    if rel_err < 5e-2:
        print("number of sum terms:", i)
        fig, ax = plt.subplots(nrows=2, ncols=1)
        im1 = ax[0].imshow(U[:,:,-1], cmap='bwr')
        plt.colorbar(im1, ax=ax[0])
        ax[0].title.set_text('SH Field')
        im2 = ax[1].imshow(np.roll(fftshift(np.real(exp_sum)),axis=(0,1),shift=(1,1)), cmap='bwr')
        plt.colorbar(im2, ax=ax[1])
        ax[1].title.set_text('Reconstruct from {} FCs'.format(i))
        plt.tight_layout()
        plt.savefig(os.getcwd() + "/Spectrum2_FourierRecon.png")
        plt.close()
        break
    i += 1

print("debug")
# min_num_periods = min(Lx,Ly)/(2*np.pi)
# max_num_periods = np.sqrt(Lx**2+Ly**2)/(2*np.pi)
# ToDo: determine window size of plane wave using wave vec diraction and dimensions of domain
est_kx = np.zeros(shape=np.shape(spec))
est_ky = np.zeros(shape=np.shape(spec))
corr_0 = np.zeros(shape=np.shape(spec))
midx = int(Nx / 2)
midy = int(Ny / 2)
num_pts_x = 16
num_pts_y = 8
print("Total coeffs:", i)
start = time.time()
for k in range(i):
    pw = np.real((FourierCoeffs[k] * np.exp(1j * (Kxs[k] * X + Kys[k] * Y))) / (Nx * Ny))
    corr_next = signal.correlate2d(U[:, :, -1],
                                   fftshift(pw[int(Ny / 2) - num_pts_y:int(Ny / 2) + num_pts_y,
                                   int(Ny / 2) - num_pts_x:int(Ny / 2) + num_pts_x]),
                                   boundary='wrap', mode='same')
    est_kx[np.where(corr_next > corr_0)] = Kxs[k]
    est_ky[np.where(corr_next > corr_0)] = Kys[k]
    if k % 100 == 0:
        print("kx = ", Kxs[k], "Ky = ",Kys[k])
    corr_0 = corr_next
end = time.time()
print("Time to perform ellipse cross correlations:",end-start)

fig, ax = plt.subplots(nrows=2, ncols=2)
im1 = ax[0, 0].imshow(est_kx, cmap='bwr')
plt.colorbar(im1, ax=ax[0, 0])
ax[0, 0].title.set_text('est_kx')
im2 = ax[0, 1].imshow(est_ky, cmap='bwr')
plt.colorbar(im2, ax=ax[0, 1])
ax[0, 1].title.set_text('est_ky')
im3 = ax[1, 0].imshow(np.sqrt(est_kx ** 2 + est_ky ** 2), cmap='bwr')
plt.colorbar(im3, ax=ax[1, 0])
ax[1, 0].title.set_text('est wave num')
im4 = ax[1, 1].imshow(U[:, :, -1], cmap='bwr')
plt.colorbar(im4, ax=ax[1, 1])
ax[1, 1].title.set_text('Field')
plt.tight_layout()
plt.savefig(os.getcwd() + "/Spectrum2_EstWaveVec.png")
plt.close()

f.close()