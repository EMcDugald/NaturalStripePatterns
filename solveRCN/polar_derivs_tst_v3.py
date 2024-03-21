import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

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

dtheta0dr = np.sqrt(R)*np.sin(3.*A/2.)
dtheta0_dr_approx = RadialDerivs(theta0,n=1)
# print("1st Radial Deriv L2 Err:",np.linalg.norm(dtheta0_dr_approx[:,1:-3]-dtheta0dr[:,1:-3]))
# print("1st Radial Deriv Mean Abs Err:",np.mean(np.abs(dtheta0_dr_approx[:,1:-3]-dtheta0dr[:,1:-3])))
# print("1st Radial Deriv Max Abs Err:",np.max(np.abs(dtheta0_dr_approx[:,1:-3]-dtheta0dr[:,1:-3])))
# print("1st Radial Deriv L2 Rel Err:",np.linalg.norm(dtheta0_dr_approx[:,1:-3]-dtheta0dr[:,1:-3])/np.linalg.norm(dtheta0dr[:,1:-3]))
print("1st Radial Deriv L2 Err:",np.linalg.norm(dtheta0_dr_approx[:,3:-3]-dtheta0dr[:,3:-3]))
print("1st Radial Deriv Mean Abs Err:",np.mean(np.abs(dtheta0_dr_approx[:,3:-3]-dtheta0dr[:,3:-3])))
print("1st Radial Deriv Max Abs Err:",np.max(np.abs(dtheta0_dr_approx[:,3:-3]-dtheta0dr[:,3:-3])))
print("1st Radial Deriv L2 Rel Err:",np.linalg.norm(dtheta0_dr_approx[:,3:-3]-dtheta0dr[:,3:-3])/np.linalg.norm(dtheta0dr[:,3:-3]))

fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(dtheta0dr[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(dtheta0_dr_approx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(dtheta0dr[:,1:-3]-dtheta0_dr_approx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.suptitle("dthetadr")
plt.tight_layout()
plt.show()

d2theta0dr2 = (1./(2*np.sqrt(R)))*np.sin(3.*A/2.)
d2theta0_dr2_approx = RadialDerivs(theta0,n=2)
print("2nd Radial Deriv L2 Err:",np.linalg.norm(d2theta0_dr2_approx[:,1:-3]-d2theta0dr2[:,1:-3]))
print("2nd Radial Deriv Mean Abs Err:",np.mean(np.abs(d2theta0_dr2_approx[:,1:-3]-d2theta0dr2[:,1:-3])))
print("2nd Radial Deriv Max Abs Err:",np.max(np.abs(d2theta0_dr2_approx[:,1:-3]-d2theta0dr2[:,1:-3])))
print("2nd Radial Deriv L2 Rel Err:",np.linalg.norm(d2theta0_dr2_approx[:,1:-3]-d2theta0dr2[:,1:-3])/np.linalg.norm(d2theta0dr2[:,1:-3]))
# print("2nd Radial Deriv L2 Err:",np.linalg.norm(d2theta0_dr2_approx[:,3:-3]-d2theta0dr2[:,3:-3]))
# print("2nd Radial Deriv Mean Abs Err:",np.mean(np.abs(d2theta0_dr2_approx[:,3:-3]-d2theta0dr2[:,3:-3])))
# print("2nd Radial Deriv Max Abs Err:",np.max(np.abs(d2theta0_dr2_approx[:,3:-3]-d2theta0dr2[:,3:-3])))
# print("2nd Radial Deriv L2 Rel Err:",np.linalg.norm(d2theta0_dr2_approx[:,3:-3]-d2theta0dr2[:,3:-3])/np.linalg.norm(d2theta0dr2[:,3:-3]))
fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(d2theta0dr2[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(d2theta0_dr2_approx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(d2theta0dr2[:,1:-3]-d2theta0_dr2_approx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.suptitle("d2thetadr2")
plt.tight_layout()
plt.show()

d3theta0dr3 = (-1/4)*R**(-3/2)*np.sin(3*A/2)
d3theta0_dr3_approx = RadialDerivs(theta0,n=3)
print("3rd Radial Deriv L2 Err:",np.linalg.norm(d3theta0_dr3_approx[:,1:-3]-d3theta0dr3[:,1:-3]))
print("3rd Radial Deriv Mean Abs Err:",np.mean(np.abs(d3theta0_dr3_approx[:,1:-3]-d3theta0dr3[:,1:-3])))
print("3rd Radial Deriv Max Abs Err:",np.max(np.abs(d3theta0_dr3_approx[:,1:-3]-d3theta0dr3[:,1:-3])))
print("3rd Radial Deriv L2 Rel Err:",np.linalg.norm(d3theta0_dr3_approx[:,1:-3]-d3theta0dr3[:,1:-3])/np.linalg.norm(d3theta0dr3[:,1:-3]))
# print("3rd Radial Deriv L2 Err:",np.linalg.norm(d3theta0_dr3_approx[:,3:-3]-d3theta0dr3[:,3:-3]))
# print("3rd Radial Deriv Mean Abs Err:",np.mean(np.abs(d3theta0_dr3_approx[:,3:-3]-d3theta0dr3[:,3:-3])))
# print("3rd Radial Deriv Max Abs Err:",np.max(np.abs(d3theta0_dr3_approx[:,3:-3]-d3theta0dr3[:,3:-3])))
# print("3rd Radial Deriv L2 Rel Err:",np.linalg.norm(d3theta0_dr3_approx[:,3:-3]-d3theta0dr3[:,3:-3])/np.linalg.norm(d3theta0dr3[:,3:-3]))
fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(d3theta0dr3[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(d3theta0_dr3_approx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(d3theta0dr3[:,1:-3]-d3theta0_dr3_approx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.suptitle("d3thetadr3")
plt.tight_layout()
plt.show()

d4theta0dr4 = (3/8)*R**(-5/2)*np.sin(3*A/2)
d4theta0_dr4_approx = RadialDerivs(theta0,n=4)
print("4th Radial Deriv L2 Err:",np.linalg.norm(d4theta0_dr4_approx[:,1:-3]-d4theta0dr4[:,1:-3]))
print("4th Radial Deriv Mean Abs Err:",np.mean(np.abs(d4theta0_dr4_approx[:,1:-3]-d4theta0dr4[:,1:-3])))
print("4th Radial Deriv Max Abs Err:",np.max(np.abs(d4theta0_dr4_approx[:,1:-3]-d4theta0dr4[:,1:-3])))
print("4th Radial Deriv L2 Rel Err:",np.linalg.norm(d4theta0_dr4_approx[:,1:-3]-d4theta0dr4[:,1:-3])/np.linalg.norm(d4theta0dr4[:,1:-3]))
# print("4th Radial Deriv L2 Err:",np.linalg.norm(d4theta0_dr4_approx[:,3:-3]-d4theta0dr4[:,3:-3]))
# print("4th Radial Deriv Mean Abs Err:",np.mean(np.abs(d4theta0_dr4_approx[:,3:-3]-d4theta0dr4[:,3:-3])))
# print("4th Radial Deriv Max Abs Err:",np.max(np.abs(d4theta0_dr4_approx[:,3:-3]-d4theta0dr4[:,3:-3])))
# print("4th Radial Deriv L2 Rel Err:",np.linalg.norm(d4theta0_dr4_approx[:,3:-3]-d4theta0dr4[:,3:-3])/np.linalg.norm(d4theta0dr4[:,3:-3]))
fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(d4theta0dr4[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(d4theta0_dr4_approx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(d4theta0dr4[:,1:-3]-d4theta0_dr4_approx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.suptitle("d4thetadr4")
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
plt.suptitle("dthetada")
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
plt.suptitle("d2thetada2")
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
plt.suptitle("d4thetada4")
plt.tight_layout()
plt.show()

d2thetada1dr1 = (3/2)*np.sqrt(R)*np.cos(3*A/2)
d2theta0_da1dr1_approx = RadialDerivs(AngularDerivs(theta0,n=1),n=1)
print("d2thetadr1da1 L2 Err:",np.linalg.norm(d2theta0_da1dr1_approx[:,1:-3]-d2thetada1dr1[:,1:-3]))
print("d2thetadr1da1 Mean Abs Err:",np.mean(np.abs(d2theta0_da1dr1_approx[:,1:-3]-d2thetada1dr1[:,1:-3])))
print("d2thetadr1da1 Max Abs Err:",np.max(np.abs(d2theta0_da1dr1_approx[:,1:-3]-d2thetada1dr1[:,1:-3])))
print("d2thetadr1da1 L2 Rel Err:",np.linalg.norm(d2theta0_da1dr1_approx[:,1:-3]-d2thetada1dr1[:,1:-3])/np.linalg.norm(d2thetada1dr1[:,1:-3]))
# print("d2thetadr1da1 L2 Err:",np.linalg.norm(d2theta0_da1dr1_approx[:,3:-3]-d2thetada1dr1[:,3:-3]))
# print("d2thetadr1da1 Mean Abs Err:",np.mean(np.abs(d2theta0_da1dr1_approx[:,3:-3]-d2thetada1dr1[:,3:-3])))
# print("d2thetadr1da1 Max Abs Err:",np.max(np.abs(d2theta0_da1dr1_approx[:,3:-3]-d2thetada1dr1[:,3:-3])))
# print("d2thetadr1da1 L2 Rel Err:",np.linalg.norm(d2theta0_da1dr1_approx[:,3:-3]-d2thetada1dr1[:,3:-3])/np.linalg.norm(d2thetada1dr1[:,3:-3]))
fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(d2thetada1dr1[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(d2theta0_da1dr1_approx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(d2thetada1dr1[:,1:-3]-d2theta0_da1dr1_approx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.suptitle("d2thetada1dr1")
plt.tight_layout()
plt.show()


d3thetada2dr1 = (-9/4)*np.sqrt(R)*np.sin(3*A/2)
d3theta0_da2dr1_approx = RadialDerivs(AngularDerivs(theta0,n=2),n=1)
print("d3thetada2dr1 L2 Err:",np.linalg.norm(d3theta0_da2dr1_approx[:,1:-3]-d3thetada2dr1[:,1:-3]))
print("d3thetada2dr1 Mean Abs Err:",np.mean(np.abs(d3theta0_da2dr1_approx[:,1:-3]-d3thetada2dr1[:,1:-3])))
print("d3thetada2dr1 Max Abs Err:",np.max(np.abs(d3theta0_da2dr1_approx[:,1:-3]-d3thetada2dr1[:,1:-3])))
print("d3thetada2dr1 L2 Rel Err:",np.linalg.norm(d3theta0_da2dr1_approx[:,1:-3]-d3thetada2dr1[:,1:-3])/np.linalg.norm(d3thetada2dr1[:,1:-3]))
# print("d3thetada2dr1 L2 Err:",np.linalg.norm(d3theta0_da2dr1_approx[:,3:-3]-d3thetada2dr1[:,3:-3]))
# print("d3thetada2dr1 Mean Abs Err:",np.mean(np.abs(d3theta0_da2dr1_approx[:,3:-3]-d3thetada2dr1[:,3:-3])))
# print("d3thetada2dr1 Max Abs Err:",np.max(np.abs(d3theta0_da2dr1_approx[:,3:-3]-d3thetada2dr1[:,3:-3])))
# print("d3thetada2dr1 L2 Rel Err:",np.linalg.norm(d3theta0_da2dr1_approx[:,3:-3]-d3thetada2dr1[:,3:-3])/np.linalg.norm(d3thetada2dr1[:,3:-3]))
fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(d3thetada2dr1[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(d3theta0_da2dr1_approx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(d3thetada2dr1[:,1:-3]-d3theta0_da2dr1_approx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.suptitle("d3thetada2dr1")
plt.tight_layout()
plt.show()


d4thetada2dr2 = (-9/8)*(R**(-1/2))*np.sin(3*A/2)
d4theta0_da2dr2_approx = RadialDerivs(AngularDerivs(theta0,n=2),n=2)
print("d4thetada2dr2 L2 Err:",np.linalg.norm(d4theta0_da2dr2_approx[:,1:-3]-d4thetada2dr2[:,1:-3]))
print("d4thetada2dr2 Mean Abs Err:",np.mean(np.abs(d4theta0_da2dr2_approx[:,1:-3]-d4thetada2dr2[:,1:-3])))
print("d4thetada2dr2 Max Abs Err:",np.max(np.abs(d4theta0_da2dr2_approx[:,1:-3]-d4thetada2dr2[:,1:-3])))
print("d4thetada2dr2 L2 Rel Err:",np.linalg.norm(d4theta0_da2dr2_approx[:,1:-3]-d4thetada2dr2[:,1:-3])/np.linalg.norm(d4thetada2dr2[:,1:-3]))
# print("d4thetada2dr2 L2 Err:",np.linalg.norm(d4theta0_da2dr2_approx[:,3:-3]-d4thetada2dr2[:,3:-3]))
# print("d4thetada2dr2 Mean Abs Err:",np.mean(np.abs(d4theta0_da2dr2_approx[:,3:-3]-d4thetada2dr2[:,3:-3])))
# print("d4thetada2dr2 Max Abs Err:",np.max(np.abs(d4theta0_da2dr2_approx[:,3:-3]-d4thetada2dr2[:,3:-3])))
# print("d4thetada2dr2 L2 Rel Err:",np.linalg.norm(d4theta0_da2dr2_approx[:,3:-3]-d4thetada2dr2[:,3:-3])/np.linalg.norm(d4thetada2dr2[:,3:-3]))
fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(d4thetada2dr2[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(d4theta0_da2dr2_approx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(d4thetada2dr2[:,1:-3]-d4theta0_da2dr2_approx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.suptitle("d4thetada2dr2")
plt.tight_layout()
plt.show()

RHS = 2*np.sqrt(R)*np.sin(3*A/2)
RHS_aprx = -(4/R**4)*d2theta0_dalpha2_approx - (2/R**2)*d2theta0_dalpha2_approx + (6/R**4)*(dtheta0_dalpha_approx**2)*d2theta0_dalpha2_approx - \
           (1/R**4)*d4theta0_dalpha4_approx - (1/R**3)*dtheta0_dr_approx - (2/R)*dtheta0_dr_approx - \
           (2/R**3)*(dtheta0_dalpha_approx**2)*dtheta0_dr_approx + (2/R**2)*d2theta0_dalpha2_approx*(dtheta0_dr_approx**2) + \
           (2/R)*dtheta0_dr_approx**3 + (8/R**2)*dtheta0_dalpha_approx*dtheta0_dr_approx*d2theta0_da1dr1_approx + \
           (2/R**3)*d3theta0_da2dr1_approx -2*d2theta0_dr2_approx + (1/R**2)*d2theta0_dr2_approx + \
           (2/R**2)*(dtheta0_dalpha_approx**2)*d2theta0_dr2_approx + 6*(dtheta0_dr_approx**2)*d2theta0_dr2_approx - \
           (2/R**2)*d4theta0_da2dr2_approx - (2/R)*d3theta0_dr3_approx - d4theta0_dr4_approx
print("RHS L2 Err:",np.linalg.norm(RHS_aprx[:,1:-3]-RHS[:,1:-3]))
print("RHS Mean Abs Err:",np.mean(np.abs(RHS_aprx[:,1:-3]-RHS[:,1:-3])))
print("RHS Max Abs Err:",np.max(np.abs(RHS_aprx[:,1:-3]-RHS[:,1:-3])))
print("RHS L2 Rel Err:",np.linalg.norm(RHS_aprx[:,1:-3]-RHS[:,1:-3])/np.linalg.norm(RHS[:,1:-3]))
# print("RHS L2 Err:",np.linalg.norm(RHS_aprx[:,3:-3]-RHS[:,3:-3]))
# print("RHS Mean Abs Err:",np.mean(np.abs(RHS_aprx[:,3:-3]-RHS[:,3:-3])))
# print("RHS Max Abs Err:",np.max(np.abs(RHS_aprx[:,3:-3]-RHS[:,3:-3])))
# print("RHS L2 Rel Err:",np.linalg.norm(RHS_aprx[:,3:-3]-RHS[:,3:-3])/np.linalg.norm(RHS[:,3:-3]))
fig, ax = plt.subplots(nrows=1,ncols=3)
im0 = ax[0].imshow(RHS[:,1:-3])
plt.colorbar(im0,ax=ax[0])
im1 = ax[1].imshow(RHS_aprx[:,1:-3])
plt.colorbar(im1,ax=ax[1])
im2 = ax[2].imshow(np.abs(RHS[:,1:-3]-RHS_aprx[:,1:-3]))
plt.colorbar(im2,ax=ax[2])
plt.suptitle("RHS")
plt.tight_layout()
plt.show()

RHS_aprx[:,[-3]] = theta0[:,[-3]]
RHS_aprx[:,[-2]] = 2*dr*dtheta0_dr_approx[:,[-3]]+RHS_aprx[:,[-4]]
RHS_aprx[:,[-1]] = 4*dr*dtheta0_dr_approx[:,[-3]]+RHS_aprx[:,[-5]]
print("debug")
RHS_aprx[0:int(na/2),[0]] = np.mean(RHS_aprx[0:int(na/2),[1]])
RHS_aprx[int(na/2):,[0]] = np.mean(RHS_aprx[int(na/2):,[1]])



#################################################################################################################
#SOME TESTS...

tstfunc = R*np.sin(3*A/2)*np.cos(.25*R)
tstd1 = np.cos(.25*R)*np.sin(3*A/2)-.25*R*np.sin(3*A/2)*np.sin(.25*R)
tstd2 = np.sin(3*A/2)*(-.0625*R*np.cos(.25*R)-.5*np.sin(.25*R))
tstd3 = np.sin(3*A/2)*(-.1875*np.cos(.25*R)+.015625*R*np.sin(.25*R))
tstd4 = np.sin(3*A/2)*(.00390625*R*np.cos(.25*R)+.0625*np.sin(.25*R))

tstd1_approx = RadialDerivs(tstfunc,n=1)
print("1st Radial Deriv L2 Err:",np.linalg.norm(tstd1_approx[:,1:-3]-tstd1[:,1:-3]))
print("1st Radial Deriv Mean Abs Err:",np.mean(np.abs(tstd1_approx[:,1:-3]-tstd1[:,1:-3])))
print("1st Radial Deriv Max Abs Err:",np.max(np.abs(tstd1_approx[:,1:-3]-tstd1[:,1:-3])))
print("1st Radial Deriv L2 Rel Err:",np.linalg.norm(tstd1_approx[:,1:-3]-tstd1[:,1:-3])/np.linalg.norm(tstd1[:,1:-3]))
print("dr^2",dr**2)

tstd2_approx = RadialDerivs(tstfunc,n=2)
print("2nd Radial Deriv L2 Err:",np.linalg.norm(tstd2_approx[:,1:-3]-tstd2[:,1:-3]))
print("2nd Radial Deriv Mean Abs Err:",np.mean(np.abs(tstd2_approx[:,1:-3]-tstd2[:,1:-3])))
print("2nd Radial Deriv Max Abs Err:",np.max(np.abs(tstd2_approx[:,1:-3]-tstd2[:,1:-3])))
print("2nd Radial Deriv L2 Rel Err:",np.linalg.norm(tstd2_approx[:,1:-3]-tstd2[:,1:-3])/np.linalg.norm(tstd2[:,1:-3]))
print("dr^2",dr**2)

tstd3_approx = RadialDerivs(tstfunc,n=3)
print("3rd Radial Deriv L2 Err:",np.linalg.norm(tstd3_approx[:,1:-3]-tstd3[:,1:-3]))
print("3rd Radial Deriv Mean Abs Err:",np.mean(np.abs(tstd3_approx[:,1:-3]-tstd3[:,1:-3])))
print("3rd Radial Deriv Max Abs Err:",np.max(np.abs(tstd3_approx[:,1:-3]-tstd3[:,1:-3])))
print("3rd Radial Deriv L2 Rel Err:",np.linalg.norm(tstd3_approx[:,1:-3]-tstd3[:,1:-3])/np.linalg.norm(tstd3[:,1:-3]))
print("dr^2",dr**2)

tstd4_approx = RadialDerivs(tstfunc,n=4)
print("4th Radial Deriv L2 Err:",np.linalg.norm(tstd4_approx[:,1:-3]-tstd4[:,1:-3]))
print("4th Radial Deriv Mean Abs Err:",np.mean(np.abs(tstd4_approx[:,1:-3]-tstd4[:,1:-3])))
print("4th Radial Deriv Max Abs Err:",np.max(np.abs(tstd4_approx[:,1:-3]-tstd4[:,1:-3])))
print("4th Radial Deriv L2 Rel Err:",np.linalg.norm(tstd4_approx[:,1:-3]-tstd4[:,1:-3])/np.linalg.norm(tstd4[:,1:-3]))
print("dr^2",dr**2)



tstfunc2 = R*np.sin(3*A/2)*np.cos(2*R)
tstd12 = np.cos(2*R)*np.sin(3*A/2)-2*R*np.sin(3*A/2)*np.sin(2*R)
tstd22 = np.sin(3*A/2)*(-4*R*np.cos(2*R)-4*np.sin(2*R))
tstd32 = np.sin(3*A/2)*(-12*np.cos(2*R)+8*R*np.sin(2*R))
tstd42 = np.sin(3*A/2)*(16*R*np.cos(2*R)+32*np.sin(2*R))

print("Larger Test Func Here")

tstd12_approx = RadialDerivs(tstfunc2,n=1)
print("1st Radial Deriv L2 Err:",np.linalg.norm(tstd12_approx[:,1:-3]-tstd12[:,1:-3]))
print("1st Radial Deriv Mean Abs Err:",np.mean(np.abs(tstd12_approx[:,1:-3]-tstd12[:,1:-3])))
print("1st Radial Deriv Max Abs Err:",np.max(np.abs(tstd12_approx[:,1:-3]-tstd12[:,1:-3])))
print("1st Radial Deriv L2 Rel Err:",np.linalg.norm(tstd12_approx[:,1:-3]-tstd12[:,1:-3])/np.linalg.norm(tstd12[:,1:-3]))
print("dr^2",dr**2)

tstd22_approx = RadialDerivs(tstfunc2,n=2)
print("2nd Radial Deriv L2 Err:",np.linalg.norm(tstd22_approx[:,1:-3]-tstd22[:,1:-3]))
print("2nd Radial Deriv Mean Abs Err:",np.mean(np.abs(tstd22_approx[:,1:-3]-tstd22[:,1:-3])))
print("2nd Radial Deriv Max Abs Err:",np.max(np.abs(tstd22_approx[:,1:-3]-tstd22[:,1:-3])))
print("2nd Radial Deriv L2 Rel Err:",np.linalg.norm(tstd22_approx[:,1:-3]-tstd22[:,1:-3])/np.linalg.norm(tstd22[:,1:-3]))
print("dr^2",dr**2)

tstd32_approx = RadialDerivs(tstfunc2,n=3)
print("3rd Radial Deriv L2 Err:",np.linalg.norm(tstd32_approx[:,1:-3]-tstd32[:,1:-3]))
print("3rd Radial Deriv Mean Abs Err:",np.mean(np.abs(tstd32_approx[:,1:-3]-tstd32[:,1:-3])))
print("3rd Radial Deriv Max Abs Err:",np.max(np.abs(tstd32_approx[:,1:-3]-tstd32[:,1:-3])))
print("3rd Radial Deriv L2 Rel Err:",np.linalg.norm(tstd32_approx[:,1:-3]-tstd32[:,1:-3])/np.linalg.norm(tstd32[:,1:-3]))
print("dr^2",dr**2)

tstd42_approx = RadialDerivs(tstfunc2,n=4)
print("4th Radial Deriv L2 Err:",np.linalg.norm(tstd42_approx[:,1:-3]-tstd42[:,1:-3]))
print("4th Radial Deriv Mean Abs Err:",np.mean(np.abs(tstd42_approx[:,1:-3]-tstd42[:,1:-3])))
print("4th Radial Deriv Max Abs Err:",np.max(np.abs(tstd42_approx[:,1:-3]-tstd42[:,1:-3])))
print("4th Radial Deriv L2 Rel Err:",np.linalg.norm(tstd42_approx[:,1:-3]-tstd42[:,1:-3])/np.linalg.norm(tstd42[:,1:-3]))
print("dr^2",dr**2)


