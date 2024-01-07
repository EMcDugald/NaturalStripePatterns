import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
import time
import math

"""
Generates Swift-Hohenberg field corresponding to "Knee Bend" solutions of RCN.
We take a preferred wave number of 1, paramaterized by 0<mu<1
The phase surface is given by sqrt(1-mu^2)x + log(2cosh(mu y))
We use RK4ETD solver, with boundary conditions imposed on the top and bottom of the rectangle
(sides where the wave vector is constant). We use an ansatz a1cos(theta)+a3cos(3theta) to impose the boundaries,
with a1,a3 found analytically. We also use this as the initial condition.
For each mu, we run the solver to T=20000, and save the data 
"""

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

def solve_sh_zigzag(Nx, mu, tmax, R, h):
    k1 = np.sqrt(1 - mu ** 2)
    k2 = mu
    Lx = 16*np.pi/k1
    Ny = 16*Nx
    y_mult = math.floor(1000 / (2 * np.pi / k2))
    Ly = y_mult*2*np.pi/k2
    yl = -Ly / 4
    yr = Ly / 4
    print("rectangle dim:", Lx, Ly)
    xx = (Lx / Nx) * np.linspace(0, Nx, Nx)
    yy = (Ly / Ny) * np.linspace(-Ny / 2 + 1, Ny / 2, Ny)
    X, Y = np.meshgrid(xx, yy)
    R = R * np.ones((Ny, Nx))

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

    hat = ((np.tanh(Y - yl) - np.tanh(Y - yr))) / 2
    u0 = np.cos(k1*X+k2*Y)*hat - np.cos(k1*X-k2*Y)*(1-hat)
    Rhat = fft2(R)
    Rhat[ind] = 0
    R = np.real(ifft2(Rhat))
    vv = fft2(u0)
    vv[ind] = 0
    u0 = np.real(ifft2(vv))
    Q[ind] = 0  # Q is the only term the multiplies the non linear factors

    nmax = int(np.round(tmax/h))
    start = time.time()
    print("starting time integration for mu=",mu)
    print("total steps:",nmax)
    for n in range(1, int(nmax) + 1):
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
        if n%20 == 0:
            print("on step:",n)
            print("energy:",np.sum(edensity(xi,eta,u0,ind,R)))


    end = time.time()
    print("time to generate solutions for mu="+str(mu)+": ", end - start)
    return u0, X, Y

















