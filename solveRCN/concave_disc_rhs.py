import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
import time

# VARIABLES AND GEOMETRY #
Lr = 10*np.pi
La = 4*np.pi
nr = 128
na = 256
r = np.linspace(0,Lr-Lr/nr,nr)
a = np.linspace(0,La-La/na,na)
R,A = np.meshgrid(r,a)
X0 = R*np.cos(A)
Y0 = R*np.sin(A)

# INITIAL CONDITION
theta_0 = ((2./3.)*R**(3./2.))*np.sin((3./2.)*A)

# REUSED QUANTITIES
dr = r[1]-r[0]
Rint = R[:,4:-4]
r_extended = r + r[-1]+dr
r_big = np.hstack((r,r_extended))
Lrbig = r_big[-1]-r_big[0]
nrbig = len(r_big)
kr = (2.*np.pi/Lrbig)*fftfreq(nrbig,1./nrbig)
ka = (2.*np.pi/La)*fftfreq(na,1./na)
Kr, Ka = np.meshgrid(kr, ka)


def FiniteDiff(arr,ord=1):
    if ord==1:
        return (-arr[:,4:]+8*arr[:,3:-1]-8*arr[:,1:-3]+arr[:,:-4])/(12*dr)
    elif ord==2:
        return (-arr[:,4:]+16*arr[:,3:-1]+-30*arr[:,2:-2]+16*arr[:,1:-3]-arr[:,:-4])/(12*(dr**2))
    else:
        raise Exception("Incompatible Order Selection")

def SpectralDiff(arr,ord=1):
    arr_refl = np.zeros((na,2*nr))
    arr_refl[:,0:nr] += arr
    arr_refl[:,nr:] += np.flip(arr,1)
    spec_refl = np.real(ifft2(((1j * Ka)**ord) * fft2(arr_refl)))
    return spec_refl[:,0:nr]

def RHSLin(arr):
    terms = SpatialDerivs(arr,type='Linear')
    coeff_r = -1/Rint**3 - 2/Rint
    coeff_rr = 1/Rint**2-2
    coeff_rrr = -2/Rint
    coeff_rrrr = -1
    coeff_aa = -4/Rint**4-2/Rint**2
    coeff_aaaa = -1/Rint**4
    coeff_aar = 2/Rint**3
    coeff_aarr = -2/Rint**2
    return coeff_r*terms[0]+coeff_rr*terms[1]+coeff_rrr*terms[2]\
           +coeff_rrrr*terms[3]+coeff_aa*terms[4]+coeff_aaaa*terms[5]\
           +coeff_aar*terms[6]+coeff_aarr*terms[7]

def RHSNonLin(arr):
    theta_r, theta_rr, theta_a, theta_aa, theta_ar = SpatialDerivs(arr,type='NonLinear')
    nlt1 = (6/Rint**4)*(theta_a**2)*(theta_aa)
    nlt2 = -(2/Rint**3)*(theta_a**2)*(theta_r)
    nlt3 = (2/Rint**2)*(theta_aa)*(theta_r)**2
    nlt4 = (2/Rint)*(theta_r)**3
    nlt5 = (8/Rint**2)*theta_a*theta_r*theta_ar
    nlt6 = (2/Rint**2)*theta_a**2*theta_rr
    nlt7 = 6*(theta_r**2)*theta_rr
    return nlt1 + nlt2 + nlt3 + nlt4 + nlt5 + nlt6 + nlt7

def RHS(arr):
    return RHSLin(arr)+RHSNonLin(arr)

def SpatialDerivs(arr,type=None):
    theta_r = FiniteDiff(arr,ord=1)[:,2:-2]
    theta_rr = FiniteDiff(arr,ord=2)[:,2:-2]
    theta_rrr = FiniteDiff(FiniteDiff(arr,ord=1),ord=2)
    theta_rrrr = FiniteDiff(FiniteDiff(arr,ord=1),ord=2)
    theta_a = SpectralDiff(arr,ord=1)[:,4:-4]
    theta_aa = SpectralDiff(arr,ord=2)[:,4:-4]
    theta_aaaa = SpectralDiff(arr,ord=4)[:,4:-4]
    theta_ar = FiniteDiff(SpectralDiff(arr,ord=1),ord=1)[:,2:-2]
    theta_aar = FiniteDiff(SpectralDiff(arr,ord=2),ord=1)[:,2:-2]
    theta_aarr = FiniteDiff(SpectralDiff(arr,ord=2),ord=2)[:,2:-2]
    if type == 'Linear':
        return theta_r, theta_rr, theta_rrr, theta_rrrr, theta_aa, theta_aaaa, theta_aar, theta_aarr
    elif type == 'NonLinear':
        return theta_r, theta_rr, theta_a, theta_aa, theta_ar
    else:
        return theta_r, theta_rr, theta_rrr, theta_rrrr, theta_a, theta_aa, theta_aaaa, theta_ar, theta_aar, theta_aarr

# n_mask = 32
# r_interior = r[n_mask:-n_mask]
r_interior = r[4:-4]
R_interior, A_interior = np.meshgrid(r_interior,a)
exact_rhs = 2*np.sqrt(R_interior)*np.sin(3*A_interior/2)

start = time.time()
#approx_rhs = RHS(theta_0)[:,int(n_mask-4):-int(n_mask-4)]
approx_rhs = RHS(theta_0)
end = time.time()
print("Time to compute RHS:",end-start)

nr_int, na_int = np.shape(R_interior)
X_interior = R_interior*np.cos(A_interior)
Y_interior = R_interior*np.sin(A_interior)
print("Max RHS Err:", np.max(np.abs(exact_rhs-approx_rhs)))
print("Mean RHS Err:", np.mean(np.abs(exact_rhs-approx_rhs)))

fig, axs = plt.subplots(nrows=2,ncols=2)
im0 = axs[0,0].scatter(X_interior[0:int(na_int/2),:],Y_interior[0:int(na_int/2),:],c=approx_rhs[0:int(na_int/2),:])
im1 = axs[1,0].scatter(X_interior[int(na_int/2):na_int,:],Y_interior[int(na_int/2):na_int,:],c=approx_rhs[int(na_int/2):na_int,:])
im2 = axs[0,1].scatter(X_interior[0:int(na_int/2),:],Y_interior[0:int(na_int/2),:],c=exact_rhs[0:int(na_int/2),:])
im3 = axs[1,1].scatter(X_interior[int(na_int/2):na_int,:],Y_interior[int(na_int/2):na_int,:],c=exact_rhs[int(na_int/2):na_int,:])
plt.colorbar(im0,ax=axs[0,0])
plt.colorbar(im1,ax=axs[1,0])
plt.colorbar(im2,ax=axs[0,1])
plt.colorbar(im3,ax=axs[1,1])
plt.suptitle("approx_rhs v exact_rhs cartestion")
plt.show()

fig, axs = plt.subplots(nrows=2,ncols=2)
im0 = axs[0,0].scatter(R_interior[0:int(na_int/2),:],A_interior[0:int(na_int/2),:],c=approx_rhs[0:int(na_int/2),:])
im1 = axs[1,0].scatter(R_interior[int(na_int/2):na_int,:],A_interior[int(na_int/2):na_int,:],c=approx_rhs[int(na_int/2):na_int,:])
im2 = axs[0,1].scatter(R_interior[0:int(na_int/2),:],A_interior[0:int(na_int/2),:],c=exact_rhs[0:int(na_int/2),:])
im3 = axs[1,1].scatter(R_interior[int(na_int/2):na_int,:],A_interior[int(na_int/2):na_int,:],c=exact_rhs[int(na_int/2):na_int,:])
plt.colorbar(im0,ax=axs[0,0])
plt.colorbar(im1,ax=axs[1,0])
plt.colorbar(im2,ax=axs[0,1])
plt.colorbar(im3,ax=axs[1,1])
plt.suptitle("approx_rhs v exact_rhs polar")
plt.show()

### Making the full fields:

rhs_exact_full = 2*np.sqrt(R)*np.sin(3*A/2)
rhs_approx_full = np.zeros(np.shape(R))
rhs_approx_full[:,4:-4] += approx_rhs
rhs_approx_full[:,int(nr-4):int(nr)] += theta_0[:,int(nr-4):int(nr)]
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X0[0:int(na/2),:],Y0[0:int(na/2),:],c=rhs_approx_full[0:int(na/2),:])
im1 = axs[1].scatter(X0[int(na/2):int(na),:],Y0[int(na/2):int(na),:],c=rhs_approx_full[int(na/2):int(na),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Full RHS Approx Cartesian")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R[0:int(na/2),:],A[0:int(na/2),:],c=rhs_approx_full[0:int(na/2),:])
im1 = axs[1].scatter(R[int(na/2):int(na),:],A[int(na/2):int(na),:],c=rhs_approx_full[int(na/2):int(na),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Full RHS Approx Polar")
plt.tight_layout()
plt.show()
#

