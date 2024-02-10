import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

Lr = 10*np.pi
La = 4*np.pi
nr = 256
na = 512
r = np.linspace(0,Lr,nr)
a = np.linspace(0,La,na)
R,A = np.meshgrid(r,a)
X0 = R*np.cos(A)
Y0 = R*np.sin(A)

### PLOT THE SURFACE ###
theta = ((2./3.)*R**(3./2.))*np.sin((3./2.)*A)
fig, axs = plt.subplots(nrows=2,ncols=1)
# top plot is 0<=a<=2pi
# bottom plot is 2pi<=a<=4pi
im0 = axs[0].scatter(X0[0:int(na/2),:],Y0[0:int(na/2),:],c=theta[0:int(na/2),:])
im1 = axs[1].scatter(X0[int(na/2):na,:],Y0[int(na/2):na,:],c=theta[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Phase Surface: Cartesian View")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(nrows=2,ncols=1)
# top plot is 0<=a<=2pi
# bottom plot is 2pi<=a<=4pi
im0 = axs[0].scatter(R[0:int(na/2),:],A[0:int(na/2),:],c=theta[0:int(na/2),:])
im1 = axs[1].scatter(R[int(na/2):na,:],A[int(na/2):na,:],c=theta[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("Phase Surface: Polar View")
plt.tight_layout()
plt.show()

print("R bounds top:", R[0:int(na/2),:][0,:][0],R[0:int(na/2),:][0,:][-1])
print("A bounds top:", A[0:int(na/2),:][:,0][0],A[0:int(na/2),:][:,0][-1])
print("R bounds bottom:", R[int(na/2):na,:][0,:][0],R[int(na/2):na,:][0,:][-1])
print("A bounds bottom:", A[int(na/2):na,:][:,0][0],A[int(na/2):na,:][:,0][-1])

### COMPUTE FIRST RADIAL DERIVATIVE ###
dr = r[1]-r[0]
#dthetadr = (theta[:,2:]-theta[:,:-2])/(2*dr) #2nd order
dthetadr = (-theta[:,4:]+8*theta[:,3:-1]-8*theta[:,1:-3]+theta[:,:-4])/(12*dr) #4th order
# need to redefine r to exclude r=0 and r=10pi points
#r1 = R[:,1:-1][0,:] #2nd order
r1 = R[:,2:-2][0,:] # 4th order
R1,A1 = np.meshgrid(r1,a)
na1, nr1 = np.shape(R1)
X1 = R1*np.cos(A1)
Y1 = R1*np.sin(A1)
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X1[0:int(na1/2),:],Y1[0:int(na1/2),:],c=dthetadr[0:int(na1/2),:])
im1 = axs[1].scatter(X1[int(na1/2):int(na1),:],Y1[int(na1/2):int(na1),:],c=dthetadr[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("dfdr Cartesian")
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R1[0:int(na1/2),:],A1[0:int(na1/2),:],c=dthetadr[0:int(na1/2),:])
im1 = axs[1].scatter(R1[int(na1/2):int(na1),:],A1[int(na1/2):int(na1),:],c=dthetadr[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("dfdr Polar")
plt.tight_layout()
plt.show()
dthetadr_exact = np.sqrt(R1)*np.sin(3*A1/2)
max_dthetadr_err = np.max(np.abs(dthetadr-dthetadr_exact))
mean_dthetadr_err = np.mean(np.abs(dthetadr-dthetadr_exact))
L2_dthetadr_err = np.linalg.norm(dthetadr-dthetadr_exact)
print("radial grid spacing (and 4th power):",dr, dr**4)
print("max_dthetadr_err:",max_dthetadr_err)
print("mean_dthetadr_err:",mean_dthetadr_err)
print("L2_dthetadr_err:",L2_dthetadr_err)

### COMPUTE FIRST ANGULAR DERIVATIVE ###
# first, lets do it with finite differences
da = a[1]-a[0]
dthetada = (-theta[4:,:]+8*theta[3:-1,:]-8*theta[1:-3,:]+theta[:-4,:])/(12*da) #4th order
a1 = A[2:-2,:][:,0] # 4th order
R1,A1 = np.meshgrid(r,a1)
na1, nr1 = np.shape(R1)
X1 = R1*np.cos(A1)
Y1 = R1*np.sin(A1)
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X1[0:int(na1/2),:],Y1[0:int(na1/2),:],c=dthetada[0:int(na1/2),:])
im1 = axs[1].scatter(X1[int(na1/2):int(na1),:],Y1[int(na1/2):int(na1),:],c=dthetada[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("dfda Cartesian")
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R1[0:int(na1/2),:],A1[0:int(na1/2),:],c=dthetada[0:int(na1/2),:])
im1 = axs[1].scatter(R1[int(na1/2):int(na1),:],A1[int(na1/2):int(na1),:],c=dthetada[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("dfda Polar")
plt.tight_layout()
plt.show()
dthetada_exact = (R1**(3/2))*np.cos(3*A1/2)
max_dthetada_err = np.max(np.abs(dthetada-dthetada_exact))
mean_dthetada_err = np.mean(np.abs(dthetada-dthetada_exact))
L2_dthetada_err = np.linalg.norm(dthetada-dthetada_exact)
print("angular grid spacing (and 4th power):",da, da**4)
print("max_dthetada_err:",max_dthetada_err)
print("mean_dthetada_err:",mean_dthetada_err)
print("L2_dthetada_err:",L2_dthetada_err)


### NOW WITH A SPECTRAL METHOD ###
a_periodic_check = np.linalg.norm(theta[0,:]-theta[-1,:])
r_periodic_check = np.linalg.norm(theta[:,0]-theta[:,-1])
print("a_periodic_check:",a_periodic_check)
print("profile diff a=0, a=4pi:", theta[0,:]-theta[-1,:])
print("r_periodic_check:",r_periodic_check)
print("profile diff r=0, r=10pi:", theta[:,0]-theta[:,-1])
# theta is periodic in alpha direction, but not in r
# we will multiply by a smoothing function to cutoff the r=0,r=10pi end points
num_transition_pts = 100
dr = r[1]-r[0]
gap = num_transition_pts*dr
print("boundary gap for radial smoother",gap)
r_start = R[0,:][0]
r_end = R[0,:][-1]
left_tanh_pt = r_start+gap/2
right_tanh_pt = r_end-gap/2
tanh_scale = 5
smoother = (np.tanh(tanh_scale*(R-left_tanh_pt))-np.tanh(tanh_scale*(R-right_tanh_pt)))/2
smoothed_theta = smoother*theta
fig, axs = plt.subplots(nrows=3,ncols=1)
im0 = axs[0].scatter(R,A,c=theta)
im1 = axs[1].scatter(R,A,c=smoother)
im2 = axs[2].scatter(R,A,c=smoothed_theta)
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.colorbar(im2,ax=axs[2])
plt.suptitle("smoothing theta")
plt.tight_layout()
plt.show()

r_start_idx_for_deriv = np.where(smoother[0,:]==1)[0][0]
r_end_idx_for_deriv = np.where(smoother[0,:]==1)[0][-1]
a_periodic_check = np.linalg.norm(smoothed_theta[0,:]-smoothed_theta[-1,:])
r_periodic_check = np.linalg.norm(smoothed_theta[:,0]-smoothed_theta[:,-1])
print("r_start_val:",r[r_start_idx_for_deriv])
print("r_end_val:",r[r_end_idx_for_deriv])
print("min smoother:",np.min(smoother))
print("max smoother:",np.max(smoother))
print("a_periodic_check for smoothed:",a_periodic_check)
print("r_periodic_check for smoothed:",r_periodic_check)

#starting the differentiation
kr = (2.*np.pi/Lr)*fftfreq(nr,1./nr)
ka = (2.*np.pi/La)*fftfreq(na,1./na)
Kr, Ka = np.meshgrid(kr, ka)
dthetada_spec = np.real(ifft2(1j*Ka*fft2(smoothed_theta)))
dthetada_spec = dthetada_spec[:,r_start_idx_for_deriv:r_end_idx_for_deriv+1]
r2 = R[:,r_start_idx_for_deriv:r_end_idx_for_deriv+1][0,:]
R2,A2 = np.meshgrid(r2,a)
na2, nr2 = np.shape(R2)
X2 = R2*np.cos(A2)
Y2 = R2*np.sin(A2)
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X2[0:int(na2/2),:],Y2[0:int(na2/2),:],c=dthetada_spec[0:int(na2/2),:])
im1 = axs[1].scatter(X2[int(na2/2):na2,:],Y2[int(na2/2):na2,:],c=dthetada_spec[int(na2/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("spectral dfda Cartesian")
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R2[0:int(na2/2),:],A2[0:int(na2/2),:],c=dthetada_spec[0:int(na2/2),:])
im1 = axs[1].scatter(R2[int(na2/2):na2,:],A2[int(na2/2):na2,:],c=dthetada_spec[int(na2/2):na2,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("spectral dfda Polar")
plt.show()
dthetada_exact = (R2**(3/2))*np.cos(3*A2/2)
max_dthetada_spec_err = np.max(np.abs(dthetada_spec-dthetada_exact))
mean_dthetada_spec_err = np.mean(np.abs(dthetada_spec-dthetada_exact))
L2_dthetada_spec_err = np.linalg.norm(dthetada_spec-dthetada_exact)
print("max_dthetada_spec_err:",max_dthetada_spec_err)
print("mean_dthetada_spec_err:",mean_dthetada_spec_err)
print("L2_dthetada_spec_err:",L2_dthetada_spec_err)


