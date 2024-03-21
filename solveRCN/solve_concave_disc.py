import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scipy.integrate import odeint

La = 4*np.pi
nr = 256
na = 256
dr = 10*np.pi/(nr-3)
Lr = 10*np.pi + 3*dr
r = np.linspace(0,Lr-Lr/nr,nr)
a = np.linspace(0,La-La/na,na)
R,A = np.meshgrid(r,a)
X = R*np.cos(A)
Y = R*np.sin(A)
dr = r[1]-r[0]

def RadialDerivs(f,n=1):
    out = np.zeros(np.shape(f))
    if n==1:
        dfdr_cent = (-(1./2.)*f[:,:-2] +
                     (1./2.)*f[:,2:]) / (dr)
        dfdr_fwd = (-(3./2.)*f[:,:-2] +
                    2*f[:,1:-1] - (1./2.)*f[:,2:]) / (dr)
        out[:,1:-1] += dfdr_cent
        out[:,0:1] += dfdr_fwd[:,0:1]
        return out
    if n==2:
        d2fdr2_cent = (-(1. / 12.) * f[:, :-4] + (4. / 3.) * f[:, 1:-3] -
                     (5./2.)*f[:,2:-2] + (4./3.)*f[:,3:-1] - (1./12.)*f[:,4:]) / (dr**2)
        d2fdr2_fwd = (2*f[:,:-4] - 5*f[:,1:-3] +
                    4*f[:,2:-2] - f[:,3:-1])/(dr**2)
        out[:, 2:-2] += d2fdr2_cent
        out[:, 0:2] += d2fdr2_fwd[:, 0:2]
        return out
    if n==3:
        d3fdr3_cent = (-(1./2.) * f[:, :-4] + f[:, 1:-3] -
                       f[:, 3:-1] + (1. / 2.) * f[:, 4:]) / (dr**3)
        d3fdr3_fwd = ((-5./2.)*f[:,:-4] + 9*f[:,1:-3] - 12*f[:,2:-2] +
                      7*f[:,3:-1] - (3./2.)*f[:,4:]) / (dr**3)
        out[:, 2:-2] += d3fdr3_cent
        out[:, 0:2] += d3fdr3_fwd[:, 0:2]
        return out
    if n==4:
        d4fdr4_cent = (f[:, :-4] - 4*f[:, 1:-3] +6*f[:,2:-2] -
                       4*f[:, 3:-1] + f[:, 4:]) / (dr**4)
        d4fdr4_fwd = (3*f[:,:-5] - 14*f[:,1:-4] + 26*f[:,2:-3] -
                      24*f[:,3:-2] + 11*f[:,4:-1] -2*f[:,5:]) / (dr**4)
        out[:, 2:-2] += d4fdr4_cent
        out[:, 0:2] += d4fdr4_fwd[:, 0:2]
        return out


def AngularDerivs(arr,n=1):
    kr = (2. * np.pi / Lr) * fftfreq(nr, 1. / nr)
    ka = (2. * np.pi / La) * fftfreq(na, 1. / na)
    Kr, Ka = np.meshgrid(kr, ka)
    return np.real(ifft2(((1j * Ka)**n) * fft2(arr)))

theta0 = ((2./3.)*R**(3./2.))*np.sin((3./2.)*A)


def RHS(func):
    dfdr = RadialDerivs(func,n=1)
    dfdr[0:int(na / 2), [0]] = np.mean(dfdr[0:int(na / 2), [1]])
    dfdr[int(na / 2):, [0]] = np.mean(dfdr[int(na / 2):, [1]])
    d2fdr2 = RadialDerivs(func,n=2)
    d2fdr2[0:int(na / 2), [0]] = np.mean(d2fdr2[0:int(na / 2), [1]])
    d2fdr2[int(na / 2):, [0]] = np.mean(d2fdr2[int(na / 2):, [1]])
    d3fdr3 = RadialDerivs(func,n=3)
    d3fdr3[0:int(na / 2), [0]] = np.mean(d3fdr3[0:int(na / 2), [1]])
    d3fdr3[int(na / 2):, [0]] = np.mean(d3fdr3[int(na / 2):, [1]])
    d4fdr4 = RadialDerivs(func,n=4)
    d4fdr4[0:int(na / 2), [0]] = np.mean(d4fdr4[0:int(na / 2), [1]])
    d4fdr4[int(na / 2):, [0]] = np.mean(d4fdr4[int(na / 2):, [1]])
    dfda = AngularDerivs(func,n=1)
    dfda[0:int(na / 2), [0]] = np.mean(dfda[0:int(na / 2), [1]])
    dfda[int(na / 2):, [0]] = np.mean(dfda[int(na / 2):, [1]])
    d2fda2 = AngularDerivs(func,n=2)
    d2fda2[0:int(na / 2), [0]] = np.mean(d2fda2[0:int(na / 2), [1]])
    d2fda2[int(na / 2):, [0]] = np.mean(d2fda2[int(na / 2):, [1]])
    d4fda4 = AngularDerivs(func,n=4)
    d4fda4[0:int(na / 2), [0]] = np.mean(d4fda4[0:int(na / 2), [1]])
    d4fda4[int(na / 2):, [0]] = np.mean(d4fda4[int(na / 2):, [1]])
    d2fdar = RadialDerivs(AngularDerivs(func,n=1),n=1)
    d2fdar[0:int(na / 2), [0]] = np.mean(d2fdar[0:int(na / 2), [1]])
    d2fdar[int(na / 2):, [0]] = np.mean(d2fdar[int(na / 2):, [1]])
    d3fdaar = RadialDerivs(AngularDerivs(func,n=2),n=1)
    d3fdaar[0:int(na / 2), [0]] = np.mean(d3fdaar[0:int(na / 2), [1]])
    d3fdaar[int(na / 2):, [0]] = np.mean(d3fdaar[int(na / 2):, [1]])
    d4fdaarr = RadialDerivs(AngularDerivs(func,n=2),n=2)
    d4fdaarr[0:int(na / 2), [0]] = np.mean(d4fdaarr[0:int(na / 2), [1]])
    d4fdaarr[int(na / 2):, [0]] = np.mean(d4fdaarr[int(na / 2):, [1]])
    RHS = -(4 / R ** 4) * d2fda2 - (2 / R ** 2) * d2fda2 + (6 / R ** 4) * (
                dfda ** 2) * d2fda2 - \
               (1 / R ** 4) * d4fda4 - (1 / R ** 3) * dfdr - (2 / R) * dfdr - \
               (2 / R ** 3) * (dfda ** 2) * dfdr + (
                           2 / R ** 2) * d2fda2 * (dfdr ** 2) + \
               (2 / R) * dfdr ** 3 + (
                           8 / R ** 2) * dfda * dfdr * d2fdar + \
               (2 / R ** 3) * d3fdaar - 2 * d2fdr2 + (1 / R ** 2) * d2fdr2 + \
               (2 / R ** 2) * (dfda ** 2) * d2fdr2 + 6 * (
                           dfdr ** 2) * d2fdr2 - \
               (2 / R ** 2) * d4fdaarr - (2 / R) * d3fdr3 - d4fdr4
    RHS[0:int(na / 2), [0]] = np.mean(RHS[0:int(na / 2), [1]])
    RHS[int(na / 2):, [0]] = np.mean(RHS[int(na / 2):, [1]])
    RHS[:, [-3]] = theta0[:, [-3]]
    RHS[:, [-2]] = 2 * dr * dfdr[:, [-3]] + RHS[:, [-4]]
    RHS[:, [-1]] = 4 * dr * dfdr[:, [-3]] + RHS[:, [-5]]
    return RHS



RHS_exact = 2*np.sqrt(R)*np.sin(3*A/2)
RHS_aprx = RHS(theta0)
print("RHS L2 Err:",np.linalg.norm(RHS_aprx[:,0:-3]-RHS_exact[:,0:-3]))
print("RHS Mean Abs Err:",np.mean(np.abs(RHS_aprx[:,0:-3]-RHS_exact[:,0:-3])))
print("RHS Max Abs Err:",np.max(np.abs(RHS_aprx[:,0:-3]-RHS_exact[:,0:-3])))
print("RHS L2 Rel Err:",np.linalg.norm(RHS_aprx[:,0:-3]-RHS_exact[:,0:-3])/np.linalg.norm(RHS_exact[:,0:-3]))
print("RHS L2 Err (no origin):",np.linalg.norm(RHS_aprx[:,3:-3]-RHS_exact[:,3:-3]))
print("RHS Mean Abs Err (no origin):",np.mean(np.abs(RHS_aprx[:,3:-3]-RHS_exact[:,3:-3])))
print("RHS Max Abs Err (no origin):",np.max(np.abs(RHS_aprx[:,3:-3]-RHS_exact[:,3:-3])))
print("RHS L2 Rel Err (no origin):",np.linalg.norm(RHS_aprx[:,3:-3]-RHS_exact[:,3:-3])/np.linalg.norm(RHS_exact[:,3:-3]))
fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(RHS_exact[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(RHS_aprx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(RHS_exact[:,1:-3]-RHS_aprx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.suptitle("RHS Polar")
plt.tight_layout()
plt.show()

X = R*np.cos(A)
Y = R*np.sin(A)
fig, ax = plt.subplots(nrows=2,ncols=3)
im0 = ax[0,0].scatter(X[0:int(na/2),1:-3],Y[0:int(na/2),1:-3],c=RHS_exact[0:int(na/2),1:-3])
plt.colorbar(im0,ax=ax[0,0])
im1 = ax[0,1].scatter(X[0:int(na/2),1:-3],Y[0:int(na/2),1:-3],c=RHS_aprx[0:int(na/2),1:-3])
plt.colorbar(im1,ax=ax[0,1])
im2 = ax[0,0].scatter(X[0:int(na/2),1:-3],Y[0:int(na/2),1:-3],c=np.abs(RHS_exact[0:int(na/2),1:-3]-RHS_aprx[0:int(na/2),1:-3]))
plt.colorbar(im2,ax=ax[0,2])
im3 = ax[1,0].scatter(X[int(na/2):na,1:-3],Y[int(na/2):na,1:-3],c=RHS_exact[int(na/2):na,1:-3])
plt.colorbar(im3,ax=ax[1,0])
im4 = ax[1,1].scatter(X[int(na/2):na,1:-3],Y[int(na/2):na,1:-3],c=RHS_aprx[int(na/2):na,1:-3])
plt.colorbar(im4,ax=ax[1,1])
im5 = ax[1,2].scatter(X[int(na/2):na,1:-3],Y[int(na/2):na,1:-3],c=np.abs(RHS_aprx[int(na/2):na,1:-3]-RHS_exact[int(na/2):na,1:-3]))
plt.colorbar(im5,ax=ax[1,2])
plt.suptitle("RHS Cartesian")
plt.tight_layout()
plt.show()


def fwd_euler(f,y,start,stop,n):
    h = (stop-start)/n
    i = 0
    while i <= n:
        k1 = h*f(y)
        y += k1
    return y


sol = fwd_euler(RHS,theta0,0,100,5000)
fig, ax = plt.subplots(nrows=1,ncols=2)
im0 = ax[0].scatter(X[0:int(na/2),1:-3],Y[0:int(na/2),1:-3],c=sol[0:int(na/2),1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].scatter(X[int(na/2):na,1:-3],Y[int(na/2):na,1:-3],c=sol[int(na/2):na,1:-3])
plt.colorbar(im3,ax=ax[1])
plt.suptitle("Stationaru Cartesian")
plt.tight_layout()
plt.show()






