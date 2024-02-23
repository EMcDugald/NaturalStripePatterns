import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

Lr = 10*np.pi
La = 4*np.pi
nr = 256
na = 512
r = np.linspace(0,Lr-Lr/nr,nr)
a = np.linspace(0,La-La/na,na)
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

### DEPRECATED ###
# ### COMPUTE FIRST ANGULAR DERIVATIVE ###
# # first, lets do it with finite differences
# da = a[1]-a[0]
# dthetada = (-theta[4:,:]+8*theta[3:-1,:]-8*theta[1:-3,:]+theta[:-4,:])/(12*da) #4th order
# a1 = A[2:-2,:][:,0] # 4th order
# R1,A1 = np.meshgrid(r,a1)
# na1, nr1 = np.shape(R1)
# X1 = R1*np.cos(A1)
# Y1 = R1*np.sin(A1)
# fig, axs = plt.subplots(nrows=2,ncols=1)
# im0 = axs[0].scatter(X1[0:int(na1/2),:],Y1[0:int(na1/2),:],c=dthetada[0:int(na1/2),:])
# im1 = axs[1].scatter(X1[int(na1/2):int(na1),:],Y1[int(na1/2):int(na1),:],c=dthetada[int(na1/2):int(na1),:])
# plt.colorbar(im0,ax=axs[0])
# plt.colorbar(im1,ax=axs[1])
# plt.suptitle("dfda Cartesian")
# plt.tight_layout()
# plt.show()
# fig, axs = plt.subplots(nrows=2,ncols=1)
# im0 = axs[0].scatter(R1[0:int(na1/2),:],A1[0:int(na1/2),:],c=dthetada[0:int(na1/2),:])
# im1 = axs[1].scatter(R1[int(na1/2):int(na1),:],A1[int(na1/2):int(na1),:],c=dthetada[int(na1/2):int(na1),:])
# plt.colorbar(im0,ax=axs[0])
# plt.colorbar(im1,ax=axs[1])
# plt.suptitle("dfda Polar")
# plt.tight_layout()
# plt.show()
# dthetada_exact = (R1**(3/2))*np.cos(3*A1/2)
# max_dthetada_err = np.max(np.abs(dthetada-dthetada_exact))
# mean_dthetada_err = np.mean(np.abs(dthetada-dthetada_exact))
# L2_dthetada_err = np.linalg.norm(dthetada-dthetada_exact)
# print("angular grid spacing (and 4th power):",da, da**4)
# print("max_dthetada_err:",max_dthetada_err)
# print("mean_dthetada_err:",mean_dthetada_err)
# print("L2_dthetada_err:",L2_dthetada_err)
### DEPRECATED ###


### NOW WITH A SPECTRAL METHOD VIA SMOOTHING ###
a_periodic_check = np.linalg.norm(theta[0,:]-theta[-1,:])
r_periodic_check = np.linalg.norm(theta[:,0]-theta[:,-1])
print("a_periodic_check:",a_periodic_check)
print("r_periodic_check:",r_periodic_check)
# theta is periodic in alpha direction, but not in r
# we will multiply by a smoothing function to cutoff the r=0,r=10pi end points
num_transition_pts = 6
dr = r[1]-r[0]
gap = num_transition_pts*dr
print("boundary gap for radial smoother",gap)
r_start = R[0,:][0]
r_end = R[0,:][-1]
left_tanh_pt = r_start+gap/2
right_tanh_pt = r_end-gap/2
tanh_scale = 10
smoother = (np.tanh(tanh_scale*(R-left_tanh_pt))-np.tanh(tanh_scale*(R-right_tanh_pt)))/2
smoothed_theta = smoother*theta
fig, axs = plt.subplots(nrows=1,ncols=3)
im0 = axs[0].imshow(theta)
im1 = axs[1].imshow(smoother)
im2 = axs[2].imshow(smoothed_theta)
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

### SPECTRAL METHOD VIA REFLECTION ###
theta_big = np.zeros((na,2*nr))
theta_big[:,0:nr] += theta
theta_big[:,nr:] += np.flip(theta,1)
fig, ax = plt.subplots()
im = ax.imshow(theta_big)
plt.colorbar(im,ax=ax)
plt.suptitle("reflected theta")
plt.tight_layout()
plt.show()
print("testing symmetry:",np.sum((theta_big[:,nr-1]!=theta_big[:,nr])))
dr = r[1]-r[0]
r_extended = r + r[-1]+dr
r_big = np.hstack((r,r_extended))
Lrbig = r_big[-1]-r_big[0]
nrbig = len(r_big)
kr = (2.*np.pi/Lrbig)*fftfreq(nrbig,1./nrbig)
ka = (2.*np.pi/La)*fftfreq(na,1./na)
Kr, Ka = np.meshgrid(kr, ka)
dthetada_big_spec = np.real(ifft2(1j*Ka*fft2(theta_big)))
dthetada_spec = dthetada_big_spec[:,0:nr]
#
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X0[0:int(na/2),:],Y0[0:int(na/2),:],c=dthetada_spec[0:int(na/2),:])
im1 = axs[1].scatter(X0[int(na/2):na,:],Y0[int(na/2):na,:],c=dthetada_spec[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("reflected spectral dfda Cartesian")
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R[0:int(na/2),:],A[0:int(na/2),:],c=dthetada_spec[0:int(na/2),:])
im1 = axs[1].scatter(R[int(na/2):na,:],A[int(na/2):na,:],c=dthetada_spec[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("reflected spectral dfda Polar")
plt.show()
dthetada_exact = (R**(3/2))*np.cos(3*A/2)
max_dthetada_spec_err = np.max(np.abs(dthetada_spec-dthetada_exact))
mean_dthetada_spec_err = np.mean(np.abs(dthetada_spec-dthetada_exact))
L2_dthetada_spec_err = np.linalg.norm(dthetada_spec-dthetada_exact)
print("max_dthetada_spec_err:",max_dthetada_spec_err)
print("mean_dthetada_spec_err:",mean_dthetada_spec_err)
print("L2_dthetada_spec_err:",L2_dthetada_spec_err)


### COMPUTE SECOND RADIAL DERIVATIVE ###
dr = r[1]-r[0]
d2thetadr2 = (-theta[:,4:]+16*theta[:,3:-1]+-30*theta[:,2:-2]+16*theta[:,1:-3]-theta[:,:-4])/(12*(dr**2)) #4th order
r1 = R[:,2:-2][0,:] # 4th order
R1,A1 = np.meshgrid(r1,a)
na1, nr1 = np.shape(R1)
X1 = R1*np.cos(A1)
Y1 = R1*np.sin(A1)
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X1[0:int(na1/2),:],Y1[0:int(na1/2),:],c=d2thetadr2[0:int(na1/2),:])
im1 = axs[1].scatter(X1[int(na1/2):int(na1),:],Y1[int(na1/2):int(na1),:],c=d2thetadr2[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d2fdr2 Cartesian")
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R1[0:int(na1/2),:],A1[0:int(na1/2),:],c=d2thetadr2[0:int(na1/2),:])
im1 = axs[1].scatter(R1[int(na1/2):int(na1),:],A1[int(na1/2):int(na1),:],c=d2thetadr2[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d2fdr2 Polar")
plt.tight_layout()
plt.show()
d2thetadr2_exact = 1/(2*np.sqrt(R1))*np.sin(3*A1/2)
max_d2thetadr2_err = np.max(np.abs(d2thetadr2-d2thetadr2_exact))
mean_d2thetadr2_err = np.mean(np.abs(d2thetadr2-d2thetadr2_exact))
L2_d2thetadr2 = np.linalg.norm(d2thetadr2-d2thetadr2_exact)
print("radial grid spacing (and 4th power):",dr, dr**4)
print("max_d2thetadr2_err:",max_d2thetadr2_err)
print("mean_d2thetadr2_err:",mean_d2thetadr2_err)
print("L2_d2thetadr2:",L2_d2thetadr2)

### COMPUTE SECOND ANGULAR DERIVATIVE (SPECTRAL VIA REFLECTION) ###
print("testing symmetry:",np.sum((theta_big[:,nr-1]!=theta_big[:,nr])))
d2thetada2_big_spec = np.real(ifft2((1j*Ka)**2*fft2(theta_big)))
d2thetada2_spec = d2thetada2_big_spec[:,0:nr]

fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X0[0:int(na/2),:],Y0[0:int(na/2),:],c=d2thetada2_spec[0:int(na/2),:])
im1 = axs[1].scatter(X0[int(na/2):na,:],Y0[int(na/2):na,:],c=d2thetada2_spec[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("reflected spectral d2fda2 Cartesian")
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R[0:int(na/2),:],A[0:int(na/2),:],c=d2thetada2_spec[0:int(na/2),:])
im1 = axs[1].scatter(R[int(na/2):na,:],A[int(na/2):na,:],c=d2thetada2_spec[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("reflected spectral d2fda2 Polar")
plt.show()
d2thetada2_exact = -(3/2)*(R**(3/2))*np.sin(3*A/2)
max_d2thetada2_spec_err = np.max(np.abs(d2thetada2_spec-d2thetada2_exact))
mean_d2thetada2_spec_err = np.mean(np.abs(d2thetada2_spec-d2thetada2_exact))
L2_d2thetada2_spec_err = np.linalg.norm(d2thetada2_spec-d2thetada2_exact)
print("max_d2thetada2_spec_err:",max_d2thetada2_spec_err)
print("mean_d2thetada2_spec_err:",mean_d2thetada2_spec_err)
print("L2_d2thetada2_spec_err:",L2_d2thetada2_spec_err)

### MIXED DERIVATIVE IN R AND ALPHA (SPECTRAL THAN FD) ###
d2thetadadr = (-dthetada_spec[:,4:]+8*dthetada_spec[:,3:-1]-8*dthetada_spec[:,1:-3]+dthetada_spec[:,:-4])/(12*dr) #4th order
# need to redefine r to exclude r=0 and r=10pi points
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X1[0:int(na1/2),:],Y1[0:int(na1/2),:],c=d2thetadadr[0:int(na1/2),:])
im1 = axs[1].scatter(X1[int(na1/2):int(na1),:],Y1[int(na1/2):int(na1),:],c=d2thetadadr[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d2fdadr Cartesian")
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R1[0:int(na1/2),:],A1[0:int(na1/2),:],c=d2thetadadr[0:int(na1/2),:])
im1 = axs[1].scatter(R1[int(na1/2):int(na1),:],A1[int(na1/2):int(na1),:],c=d2thetadadr[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d2fdadr Polar")
plt.tight_layout()
plt.show()
d2thetadadr_exact = (3/2)*np.sqrt(R1)*np.cos(3*A1/2)
max_d2thetadadr_err = np.max(np.abs(d2thetadadr-d2thetadadr_exact))
mean_d2thetadadr_err = np.mean(np.abs(d2thetadadr-d2thetadadr_exact))
L2_d2thetadadr_err = np.linalg.norm(d2thetadadr-d2thetadadr_exact)
print("max_d2thetadadr_err:",max_d2thetadadr_err)
print("mean_d2thetadadr_err:",mean_d2thetadadr_err)
print("L2_d2thetadadr_err:",L2_d2thetadadr_err)


### THIRD RADIAL DERIVATIVE ###
dr = r[1]-r[0]
# d3thetadr3 = (theta[:,4:]-2*theta[:,3:-1]+2*theta[:,1:-3]-theta[:,:-4])/(2*(dr**3)) #2nd order
# r1 = R[:,2:-2][0,:] # 2nd order
d3thetadr3 = (-d2thetadr2[:,4:]+8*d2thetadr2[:,3:-1]-8*d2thetadr2[:,1:-3]+d2thetadr2[:,:-4])/(12*dr)
r3 = R[:,4:-4][0,:] #4th order
R3,A3 = np.meshgrid(r3,a)
na3, nr3 = np.shape(R3)
X3 = R3*np.cos(A3)
Y3 = R3*np.sin(A3)
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X3[0:int(na3/2),:],Y3[0:int(na3/2),:],c=d3thetadr3[0:int(na3/2),:])
im1 = axs[1].scatter(X3[int(na3/2):int(na3),:],Y3[int(na3/2):int(na3),:],c=d3thetadr3[int(na3/2):int(na3),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d3thetadr3 Cartesian")
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R3[0:int(na3/2),:],A3[0:int(na3/2),:],c=d3thetadr3[0:int(na3/2),:])
im1 = axs[1].scatter(R3[int(na3/2):int(na1),:],A3[int(na3/2):int(na3),:],c=d3thetadr3[int(na3/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d3thetadr3 Polar")
plt.tight_layout()
plt.show()
d3thetadr3_exact = (-1/(4*(R3)**(3/2)))*np.sin(3*A3/2)
max_d3thetadr3_err = np.max(np.abs(d3thetadr3-d3thetadr3_exact))
mean_d3thetadr3_err = np.mean(np.abs(d3thetadr3-d3thetadr3_exact))
L2_d3thetadr3 = np.linalg.norm(d3thetadr3-d3thetadr3_exact)
print("radial grid spacing (and 4th power):",dr, dr**4)
print("max_d3thetadr3_err:",max_d3thetadr3_err)
print("mean_d3thetadr3_err:",mean_d3thetadr3_err)
print("L2_d3thetadr3:",L2_d3thetadr3)

### MIXED DERIVATIVE IN R ALPHA ALPHA (SPECTRAL THAN FD) ###
d3thetada2dr = (-d2thetada2_spec[:,4:]+8*d2thetada2_spec[:,3:-1]-8*d2thetada2_spec[:,1:-3]+d2thetada2_spec[:,:-4])/(12*dr) #4th order
# need to redefine r to exclude r=0 and r=10pi points
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X1[0:int(na1/2),:],Y1[0:int(na1/2),:],c=d2thetadadr[0:int(na1/2),:])
im1 = axs[1].scatter(X1[int(na1/2):int(na1),:],Y1[int(na1/2):int(na1),:],c=d2thetadadr[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d3thetada2dr Cartesian")
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R1[0:int(na1/2),:],A1[0:int(na1/2),:],c=d3thetada2dr[0:int(na1/2),:])
im1 = axs[1].scatter(R1[int(na1/2):int(na1),:],A1[int(na1/2):int(na1),:],c=d3thetada2dr[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d3thetada2dr Polar")
plt.tight_layout()
plt.show()
d3thetada2dr_exact = -(9/4)*np.sqrt(R1)*np.sin(3*A1/2)
max_d3thetada2dr_err = np.max(np.abs(d3thetada2dr-d3thetada2dr_exact))
mean_d3thetada2dr_err = np.mean(np.abs(d3thetada2dr-d3thetada2dr_exact))
L2_d3thetada2dr_err = np.linalg.norm(d3thetada2dr-d3thetada2dr_exact)
print("max_d3thetada2dr_err:",max_d3thetada2dr_err)
print("mean_d3thetada2dr_err:",mean_d3thetada2dr_err)
print("L2_d3thetada2dr_err:",L2_d3thetada2dr_err)


### FOURTH RADIAL DERIVATIVE ###
dr = r[1]-r[0]
d4thetadr4 = (-d2thetadr2[:,4:]+16*d2thetadr2[:,3:-1]+-30*d2thetadr2[:,2:-2]+16*d2thetadr2[:,1:-3]-d2thetadr2[:,:-4])/(12*(dr**2)) #4th order
r3 = R[:,4:-4][0,:] #4th order
R3,A3 = np.meshgrid(r3,a)
na3, nr3 = np.shape(R3)
X3 = R3*np.cos(A3)
Y3 = R3*np.sin(A3)
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X3[0:int(na3/2),:],Y3[0:int(na3/2),:],c=d4thetadr4[0:int(na3/2),:])
im1 = axs[1].scatter(X3[int(na3/2):int(na3),:],Y3[int(na3/2):int(na3),:],c=d4thetadr4[int(na3/2):int(na3),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d4thetadr4 Cartesian")
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R3[0:int(na3/2),:],A3[0:int(na3/2),:],c=d4thetadr4[0:int(na3/2),:])
im1 = axs[1].scatter(R3[int(na3/2):int(na1),:],A3[int(na3/2):int(na3),:],c=d4thetadr4[int(na3/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d4thetadr4 Polar")
plt.tight_layout()
plt.show()
d4thetadr4_exact = (3/(8*(R3)**(5/2)))*np.sin(3*A3/2)
max_d4thetadr4_err = np.max(np.abs(d4thetadr4-d4thetadr4_exact))
mean_d4thetadr4_err = np.mean(np.abs(d4thetadr4-d4thetadr4_exact))
L2_d4thetadr4 = np.linalg.norm(d4thetadr4-d4thetadr4_exact)
print("radial grid spacing (and 4th power):",dr, dr**4)
print("max_d4thetadr4_err:",max_d4thetadr4_err)
print("mean_d4thetadr4_err:",mean_d4thetadr4_err)
print("L2_d4thetadr4:",L2_d4thetadr4)


### COMPUTE FOURTH ANGULAR DERIVATIVE (SPECTRAL VIA REFLECTION) ###
print("testing symmetry:",np.sum((theta_big[:,nr-1]!=theta_big[:,nr])))
d4thetada4_big_spec = np.real(ifft2((1j*Ka)**4*fft2(theta_big)))
d4thetada4_spec = d4thetada4_big_spec[:,0:nr]

fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X0[0:int(na/2),:],Y0[0:int(na/2),:],c=d4thetada4_spec[0:int(na/2),:])
im1 = axs[1].scatter(X0[int(na/2):na,:],Y0[int(na/2):na,:],c=d4thetada4_spec[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("reflected spectral d4thetada4 Cartesian")
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R[0:int(na/2),:],A[0:int(na/2),:],c=d4thetada4_spec[0:int(na/2),:])
im1 = axs[1].scatter(R[int(na/2):na,:],A[int(na/2):na,:],c=d4thetada4_spec[int(na/2):na,:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("reflected spectral d4thetada4 Polar")
plt.show()
d4thetada4_spec_exact = (27/8)*(R**(3/2))*np.sin(3*A/2)
max_d4thetada4_spec_err = np.max(np.abs(d4thetada4_spec-d4thetada4_spec_exact))
mean_d4thetada4_spec_err = np.mean(np.abs(d4thetada4_spec-d4thetada4_spec_exact))
L2_d4thetada4_spec_err = np.linalg.norm(d4thetada4_spec-d4thetada4_spec_exact)
print("max_d4thetada4_spec_err:",max_d4thetada4_spec_err)
print("mean_d4thetada4_spec_err:",mean_d4thetada4_spec_err)
print("L2_d4thetada4_spec_err:",L2_d4thetada4_spec_err)


### MIXED DERIVATIVE IN RR ALPHA ALPHA (SPECTRAL THAN FD) ###
d4thetada2dr2 = (-d2thetada2_spec[:,4:]+16*d2thetada2_spec[:,3:-1]+-30*d2thetada2_spec[:,2:-2]+16*d2thetada2_spec[:,1:-3]-d2thetada2_spec[:,:-4])/(12*(dr**2))
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(X1[0:int(na1/2),:],Y1[0:int(na1/2),:],c=d4thetada2dr2[0:int(na1/2),:])
im1 = axs[1].scatter(X1[int(na1/2):int(na1),:],Y1[int(na1/2):int(na1),:],c=d4thetada2dr2[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d4thetada2dr2 Cartesian")
plt.tight_layout()
plt.show()
fig, axs = plt.subplots(nrows=2,ncols=1)
im0 = axs[0].scatter(R1[0:int(na1/2),:],A1[0:int(na1/2),:],c=d4thetada2dr2[0:int(na1/2),:])
im1 = axs[1].scatter(R1[int(na1/2):int(na1),:],A1[int(na1/2):int(na1),:],c=d4thetada2dr2[int(na1/2):int(na1),:])
plt.colorbar(im0,ax=axs[0])
plt.colorbar(im1,ax=axs[1])
plt.suptitle("d4thetada2dr2 Polar")
plt.tight_layout()
plt.show()
d4thetada2dr2_exact = -(9/(8*np.sqrt(R1)))*np.sin(3*A1/2)
max_d4thetada2dr2_err = np.max(np.abs(d4thetada2dr2-d4thetada2dr2_exact))
mean_d4thetada2dr2_err = np.mean(np.abs(d4thetada2dr2-d4thetada2dr2_exact))
L2_d4thetada2dr2_err = np.linalg.norm(d4thetada2dr2-d4thetada2dr2_exact)
print("max_d4thetada2dr2_err:",max_d4thetada2dr2_err)
print("mean_d4thetada2dr2_err:",mean_d4thetada2dr2_err)
print("L2_d4thetada2dr2_err:",L2_d4thetada2dr2_err)
