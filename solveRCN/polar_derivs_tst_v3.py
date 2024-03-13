import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

La = 4*np.pi
nr = 128
na = 128
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


# def SpectralDiff(arr,ord=1):
#     arr_refl = np.zeros((na,2*nr))
#     arr_refl[:,0:nr] += arr
#     arr_refl[:,nr:] += np.flip(arr,1)
#     spec_refl = np.real(ifft2(((1j * Ka)**ord) * fft2(arr_refl)))
#     return spec_refl[:,0:nr]

def AngularDerivs(arr,n=1):
    kr = (2. * np.pi / Lr) * fftfreq(nr, 1. / nr)
    ka = (2. * np.pi / La) * fftfreq(na, 1. / na)
    Kr, Ka = np.meshgrid(kr, ka)
    return np.real(ifft2(((1j * Ka)**n) * fft2(arr)))

theta0 = ((2./3.)*R**(3./2.))*np.sin((3./2.)*A)

dtheta0dr = np.sqrt(R)*np.sin(3.*A/2.)
dtheta0_dr_approx = RadialDerivs(theta0,n=1)
print("1st Radial Deriv L2 Err:",np.linalg.norm(dtheta0_dr_approx[:,1:-3]-dtheta0dr[:,1:-3]))
print("1st Radial Deriv Mean Abs Err:",np.mean(np.abs(dtheta0_dr_approx[:,1:-3]-dtheta0dr[:,1:-3])))
print("1st Radial Deriv Max Abs Err:",np.max(np.abs(dtheta0_dr_approx[:,1:-3]-dtheta0dr[:,1:-3])))

fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(dtheta0dr[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(dtheta0_dr_approx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(dtheta0dr[:,1:-3]-dtheta0_dr_approx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.tight_layout()
plt.show()

d2theta0dr2 = (1./(2*np.sqrt(R+1e-8)))*np.sin(3.*A/2.)
d2theta0_dr2_approx = RadialDerivs(theta0,n=2)
print("2nd Radial Deriv L2 Err:",np.linalg.norm(d2theta0_dr2_approx[:,1:-3]-d2theta0dr2[:,1:-3]))
print("2nd Radial Deriv Mean Abs Err:",np.mean(np.abs(d2theta0_dr2_approx[:,1:-3]-d2theta0dr2[:,1:-3])))
print("2nd Radial Deriv Max Abs Err:",np.max(np.abs(d2theta0_dr2_approx[:,1:-3]-d2theta0dr2[:,1:-3])))

fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(d2theta0dr2[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(d2theta0_dr2_approx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(d2theta0dr2[:,1:-3]-d2theta0_dr2_approx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.tight_layout()
plt.show()


d3theta0dr3 = (-1./(4*(R+1e-8)**(3./2.)))*np.sin(3.*A/2.)
d3theta0_dr3_approx = RadialDerivs(theta0,n=3)
print("3rd Radial Deriv L2 Err:",np.linalg.norm(d3theta0_dr3_approx[:,1:-3]-d3theta0dr3[:,1:-3]))
print("3rd Radial Deriv Mean Abs Err:",np.mean(np.abs(d3theta0_dr3_approx[:,1:-3]-d3theta0dr3[:,1:-3])))
print("3rd Radial Deriv Max Abs Err:",np.max(np.abs(d3theta0_dr3_approx[:,1:-3]-d3theta0dr3[:,1:-3])))

fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(d3theta0dr3[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(d3theta0_dr3_approx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(d3theta0dr3[:,1:-3]-d3theta0_dr3_approx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.tight_layout()
plt.show()

d4theta0dr4 = (1./(8*(R+1e-8)**(5./2.)))*np.sin(3.*A/2.)
d4theta0_dr4_approx = RadialDerivs(theta0,n=4)
print("4th Radial Deriv L2 Err:",np.linalg.norm(d4theta0_dr4_approx[:,1:-3]-d4theta0dr4[:,1:-3]))
print("4th Radial Deriv Mean Abs Err:",np.mean(np.abs(d4theta0_dr4_approx[:,1:-3]-d4theta0dr4[:,1:-3])))
print("4th Radial Deriv Max Abs Err:",np.max(np.abs(d4theta0_dr4_approx[:,1:-3]-d4theta0dr4[:,1:-3])))

fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(d4theta0dr4[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(d4theta0_dr4_approx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(d4theta0dr4[:,1:-3]-d4theta0_dr4_approx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.tight_layout()
plt.show()


dtheta0dalpha = R**(3./2.)*np.cos(3.*A/2.)
dtheta0_dalpha_approx = AngularDerivs(theta0,n=1)
print("1st Angular Deriv L2 Err:",np.linalg.norm(dtheta0_dalpha_approx-dtheta0dalpha))
print("1st Angular Deriv Mean Abs Err:",np.mean(np.abs(dtheta0_dalpha_approx-dtheta0dalpha)))
print("1st Angular Deriv Max Abs Err:",np.max(np.abs(dtheta0_dalpha_approx-dtheta0dalpha)))

fig, ax = plt.subplots(nrows=1,ncols=2)
im0 = ax[0].imshow(dtheta0dalpha)
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(dtheta0_dalpha_approx)
plt.colorbar(im1,ax=ax[1])
plt.tight_layout()
plt.show()


d2theta0dalpha2 = -(3./2.)*R**(3./2.)*np.sin(3.*A/2.)
d2theta0_dalpha2_approx = AngularDerivs(theta0,n=2)
print("2nd Angular Deriv L2 Err:",np.linalg.norm(d2theta0_dalpha2_approx-d2theta0dalpha2))
print("2nd Angular Deriv Mean Abs Err:",np.mean(np.abs(d2theta0_dalpha2_approx-d2theta0dalpha2)))
print("2nd Angular Deriv Max Abs Err:",np.max(np.abs(d2theta0_dalpha2_approx-d2theta0dalpha2)))

fig, ax = plt.subplots(nrows=1,ncols=2)
im0 = ax[0].imshow(d2theta0dalpha2)
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(d2theta0_dalpha2_approx)
plt.colorbar(im1,ax=ax[1])
plt.tight_layout()
plt.show()


d4theta0dalpha4 = (27./8.)*R**(3./2.)*np.sin(3.*A/2.)
d4theta0_dalpha4_approx = AngularDerivs(theta0,n=4)
print("4th Angular Deriv L2 Err:",np.linalg.norm(d4theta0_dalpha4_approx-d4theta0dalpha4))
print("4th Angular Deriv Mean Abs Err:",np.mean(np.abs(d4theta0_dalpha4_approx-d4theta0dalpha4)))
print("4th Angular Deriv Max Abs Err:",np.max(np.abs(d4theta0_dalpha4_approx-d4theta0dalpha4)))

fig, ax = plt.subplots(nrows=1,ncols=2)
im0 = ax[0].imshow(d4theta0dalpha4)
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(d4theta0_dalpha4_approx)
plt.colorbar(im1,ax=ax[1])
plt.tight_layout()
plt.show()

# print("debug")
# d2theta0dr2 = (1./(2*np.sqrt(R)))*np.sin(3.*A/2.)
# d3theta0dr3 = (-1./(4*R**(3./2.)))*np.sin(3.*A/2.)
# d4theta0dr4 = (1./(8*R**(5./2.)))*np.sin(3.*A/2.)








