import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
import time
import math
import matplotlib.pyplot as plt
import os
import scipy.io as sio

Lx = 10*np.pi
Ly = 10*np.pi
Nx = 128
Ny = 128
R = .5
tmax = 100
mu = .7
k1 = np.sqrt(1 - mu ** 2)
k2 = mu
xx = (Lx / Nx) * np.linspace(0, Nx, Nx)
yy = (Ly / Ny) * np.linspace(-Ny / 2 + 1, Ny / 2, Ny)
X, Y = np.meshgrid(xx, yy)

theta = np.sqrt(1-mu**2)*X + np.log(2*np.cosh(mu*Y))
w = np.cos(theta)
fig, ax = plt.subplots(nrows=1,ncols=1)
ax.imshow(w)
plt.suptitle("Regular Pattern")
plt.tight_layout()
plt.show()

alpha = np.pi/3
X_R = np.cos(alpha)*X - np.sin(alpha)*Y
Y_R = np.sin(alpha)*X + np.cos(alpha)*Y
theta_R = np.sqrt(1-mu**2)*X_R + np.log(2*np.cosh(mu*Y_R))
w_R = np.cos(theta_R)
fig, ax = plt.subplots(nrows=1,ncols=1)
ax.imshow(w_R)
plt.suptitle("Rotated Pattern")
plt.tight_layout()
plt.show()


m, sig = 0, 0.01
def perturbed_theta():
    mu = np.random.uniform(0,1,1)
    alpha = 2*np.pi*np.random.uniform(0,1,1)
    x = np.cos(alpha)*X - np.sin(alpha)*Y
    y = np.sin(alpha)*X + np.cos(alpha)*Y
    tst1 = np.random.uniform(0,10,1)
    if tst1 <= 3:
        yterm = 2*np.cosh(mu*y)
    else:
        yterm = 2*perturbed_cosh(mu*y)
    tst2 = np.random.uniform(0,10,1)
    if tst2 <= 3:
        xterm = x
    else:
        xterm = x + np.random.normal(m, sig, 1)
    return np.sqrt(1-mu**2)*xterm + np.log(yterm)


def perturbed_cosh(y):
    xi1 = np.random.normal(m, sig, 1)
    xi2 = np.random.normal(m, sig, 1)
    xi3 = np.random.normal(m, sig, 1)
    xi4 = np.random.normal(m, sig, 1)
    print("cosh random samples:",xi1,xi2,xi3,xi4)
    c1 = (1+xi1)*1
    c2 = (1+xi2)*(1/2)
    c3 = (1+xi3)*(1/math.factorial(4))
    c4 = (1+xi4)*(1/math.factorial(6))
    return c1 + c2*y**2 + c3*y**4 + c4*y**6


for i in range(1,10):
    print("Generating Sample Plots ",i)
    theta = perturbed_theta()
    w = np.cos(theta)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(w)
    ax[1].imshow(theta)
    plt.suptitle("Perturbed theta {}".format(i))
    plt.tight_layout()
    plt.show()


num_samples = 1500
data = np.zeros(shape=(num_samples,Ny,Nx))
labels = np.zeros(shape=(num_samples,Ny,Nx))
for i in range(num_samples):
    print("Generating Data ", i)
    theta = perturbed_theta()
    labels[i,:,:] = theta
    data[i,:,:] = np.cos(theta)

mdict = {"w": data, "phase": labels}
dir = os.getcwd()+"/data/"
sio.savemat(dir+"nn_data"+".mat", mdict)















