import numpy as np
from scipy.fft import fft2, fftfreq, fftshift
import os
import matplotlib.pyplot as plt
import sys
import time



#mu = .3
#mu = .5
#mu = .7
mu = .9

logfile = open(os.getcwd()+"/logs/kb_wv_local_{}.out".format(mu), 'w')
sys.stdout = logfile

Nx = 256
Ny = 256
k1 = np.sqrt(1-mu**2)
k2 = mu
Ly = 20*np.pi/k1
Lx = Ly
xx = (Lx/Nx)*np.linspace(-Nx/2+1,Nx/2,Nx)
yy = (Ly/Ny)*np.linspace(-Ny/2+1,Ny/2,Ny)
X,Y = np.meshgrid(xx,yy)
theta = k1*X + np.log(2*np.cosh(k2*Y))
W = np.cos(theta)

def theta_x(x,y):
    return k1*np.ones(shape=np.shape(x))

def theta_y(x,y):
    return k2*np.tanh(k2*y)

kx_exact = theta_x(X,Y)
ky_exact = theta_y(X,Y)
wave_nums_exact = np.sqrt(kx_exact**2+ky_exact**2)

kx = (2. * np.pi / Lx) * fftfreq(Nx, 1. / Nx)  # wave numbers
ky = (2. * np.pi / Ly) * fftfreq(Ny, 1. / Ny)
Kx, Ky = np.meshgrid(kx, ky)


ss_factor = 4
col_indices = np.where((X[0,:]>-Lx/4) & (X[0,:]<Lx/4))[0][::ss_factor]
row_indices = np.where((Y[:,0]>-Ly/4) & (Y[:,0]<Ly/4))[0][::ss_factor]
innerW = W[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1][::ss_factor,::ss_factor]
innerX = X[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1][::ss_factor,::ss_factor]
innerY = Y[row_indices[0]:row_indices[-1]+1,
         col_indices[0]:col_indices[-1]+1][::ss_factor,::ss_factor]
wave_nums_exact_inner = wave_nums_exact[row_indices[0]:row_indices[-1]+1,
                        col_indices[0]:col_indices[-1]+1][::ss_factor,::ss_factor]
kx_exact_inner = kx_exact[row_indices[0]:row_indices[-1]+1,
                 col_indices[0]:col_indices[-1]+1][::ss_factor,::ss_factor]
ky_exact_inner = ky_exact[row_indices[0]:row_indices[-1]+1,
                 col_indices[0]:col_indices[-1]+1][::ss_factor,::ss_factor]

kx_approx = np.zeros(np.shape(innerW))
ky_approx = np.zeros(np.shape(innerW))
wave_nums_approx = np.zeros(np.shape(innerW))


def Gaussian(x0,y0,X,Y,sigma):
    exponent = (X-x0)**2 + (Y-y0)**2
    return np.exp(-exponent/(sigma**2))

start = time.time()
r_shift = row_indices[0]
c_shift = col_indices[0]
for r in row_indices:
    for c in col_indices:
        print("getting wave vector at indices:",r,c)
        G = Gaussian(X[r, c], Y[r, c], X, Y, 3.3)
        f = G*W
        spec = fftshift(fft2(f))
        max_spec_idx = np.argsort(-np.abs(spec).flatten())[0]
        k_x = fftshift(Kx).flatten()[max_spec_idx]
        k_y = fftshift(Ky).flatten()[max_spec_idx]
        kx_approx[int((r-r_shift)/ss_factor),int((c-c_shift)/ss_factor)] += k_x
        ky_approx[int((r-r_shift)/ss_factor),int((c-c_shift)/ss_factor)] += k_y
        print("kx = ",k_x, " ky= ", k_y)
end = time.time()


wave_nums_approx = np.sqrt(kx_approx**2+ky_approx**2)
kx_approx = np.abs(kx_approx)
ky_approx = np.where(innerY>0,np.abs(ky_approx),-np.abs(ky_approx))


fig, ax = plt.subplots(figsize=(10,10))
ax.quiver(innerX.flatten(),innerY.flatten(),
          kx_approx.flatten(),ky_approx.flatten())
plt.savefig(os.getcwd()+"/figs/local_spec/"+"approx_quiver_mu_{}.png".format(mu))

fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(4,10))
im1 = ax[0].imshow(kx_approx,cmap='bwr')
im2 = ax[1].imshow(kx_exact_inner,cmap='bwr')
ax[0].title.set_text('kx_approx')
ax[1].title.set_text('kx_exact')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/local_spec/"+"kx_mu_{}.png".format(mu))
plt.close()

fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(4,10))
im1 = ax[0].imshow(ky_approx,cmap='bwr')
im2 = ax[1].imshow(ky_exact_inner,cmap='bwr')
ax[0].title.set_text('ky_approx')
ax[1].title.set_text('ky_exact')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/local_spec/"+"ky_mu_{}.png".format(mu))
plt.close()

fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(4,10))
im1 = ax[0].imshow(wave_nums_approx,cmap='bwr')
im2 = ax[1].imshow(wave_nums_exact_inner,cmap='bwr')
ax[0].title.set_text('wave_nums_approx')
ax[1].title.set_text('wave_nums_exaxt')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/local_spec/"+"wave_nums_mu_{}.png".format(mu))
plt.close()


w_hat1 = np.cos(kx_exact_inner*innerX+ky_exact_inner*innerY)
w_hat2 = np.cos(kx_approx*innerX+ky_approx*innerY)
print("time to get wave numbers:",end-start)
print("L2 Wave Num Rel Err:",np.linalg.norm(wave_nums_exact_inner-wave_nums_approx)/np.linalg.norm(wave_nums_exact_inner))
print("L2 Kx Rel Err:",np.linalg.norm(kx_approx-kx_exact_inner)/np.linalg.norm(kx_exact_inner))
print("L2 Ky Rel Err:",np.linalg.norm(ky_approx-ky_exact_inner)/np.linalg.norm(ky_exact_inner))
# print("L2 Rel Err cos(kx*x + ky*y):", np.linalg.norm(w_hat1-w_hat2)/np.linalg.norm(w_hat1))
# print("L2 Rel Err approx cos(kx*x + ky*y) vs W:", np.linalg.norm(w_hat2-innerW)/np.linalg.norm(innerW))


fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(4,14))
im1 = ax[0].imshow(w_hat1,cmap='bwr')
im2 = ax[1].imshow(w_hat2,cmap='bwr')
im3 = ax[2].imshow(innerW,cmap='bwr')
ax[0].title.set_text('exact cos(kx*x+ky*y)')
ax[1].title.set_text('approx cos(kx*x+ky*y)')
ax[2].title.set_text('W')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/local_spec/"+"linearized_phases_mu_{}.png".format(mu))
plt.close()


###### TRYING TO USE THE WAVE VECTORS AS INITIAL CONDITION... #######

def approx_field(x):
    return np.cos(x[0]*innerX+x[1]*innerY)

def F(x):
    return -((approx_field(x)-innerW)**2)

def gradF(x):
    dFdkx = -2*(np.cos(x[0]*innerX+x[1]*innerY)-innerW)*np.sin(x[0]*innerX+x[1]*innerY)*innerX
    dFdky = -2*(np.cos(x[0]*innerX+x[1]*innerY)-innerW)*np.sin(x[0]*innerX+x[1]*innerY)*innerY
    return np.array([dFdkx,dFdky])

def grad_descent_bt0(x,tol,f,gradf):
    F, dF, ctr = f(x), gradf(x), 0
    while np.linalg.norm(approx_field(x)-innerW)/np.linalg.norm(innerW) > tol:
        if ctr % 2000 == 0:
            print("rel err is:", np.linalg.norm(approx_field(x)-innerW)/np.linalg.norm(innerW))
        gamma = (.01*tol)/(np.max(np.abs(dF)))
        x = x - gamma*dF
        F = f(x)
        dF = gradf(x)
        ctr = ctr + 1
    return x, F, dF, ctr

init = np.array([kx_approx,ky_approx])
#wvs, obj, gradobj, ctr = grad_descent_bt0(init,1e-2,F,gradF)
#wvs, obj, gradobj, ctr = grad_descent_bt0(init,.019,F,gradF)
#wvs, obj, gradobj, ctr = grad_descent_bt0(init,.011,F,gradF)
wvs, obj, gradobj, ctr = grad_descent_bt0(init,.03,F,gradF)
print("number of gradient descent steps:",ctr)

wn_gd_approx = np.sqrt(wvs[0]**2+wvs[1]**2)

print("With GD- L2 Field Rel Err:",np.linalg.norm(innerW-np.cos(wvs[0]*innerX+wvs[1]*innerY))/np.linalg.norm(innerW))
print("With GD- L2 Wave Num Rel Err:",np.linalg.norm(wave_nums_exact_inner-wn_gd_approx)/np.linalg.norm(wave_nums_exact_inner))
print("With GD- Wave Num MSE:",np.mean(np.abs(wave_nums_exact_inner-wn_gd_approx)))
print("With GD- L2 Kx Rel Err:",np.linalg.norm(wvs[0]-kx_exact_inner)/np.linalg.norm(kx_exact_inner))
print("With GD- Kx MSE:",np.mean(np.abs(wvs[0]-kx_exact_inner)))
print("With GD- L2 Ky Rel Err:",np.linalg.norm(wvs[1]-ky_exact_inner)/np.linalg.norm(ky_exact_inner))
print("With GD- Ky MSE:",np.mean(np.abs(wvs[1]-ky_exact_inner)))

fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(4,14))
im1 = ax[0].imshow(innerW,cmap='bwr')
im2 = ax[1].imshow(np.cos(wvs[0]*innerX+wvs[1]*innerY),cmap='bwr')
im3 = ax[2].imshow(np.abs(innerW-np.cos(wvs[0]*innerX+wvs[1]*innerY)),cmap='bwr')
ax[0].title.set_text('exact field')
ax[1].title.set_text('GD Approximation')
ax[2].title.set_text('Abs Err')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/local_spec/"+"grad_descent_field_compare_mu_{}.png".format(mu))
plt.close()

fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(4,14))
im1 = ax[0].imshow(kx_exact_inner,cmap='bwr')
im2 = ax[1].imshow(wvs[0],cmap='bwr')
im3 = ax[2].imshow(np.abs(kx_exact_inner-wvs[0]),cmap='bwr')
ax[0].title.set_text('exact kx')
ax[1].title.set_text('GD Approximation')
ax[2].title.set_text('Abs Err')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/local_spec/"+"grad_descent_kx_compare_mu_{}.png".format(mu))
plt.close()

fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(4,14))
im1 = ax[0].imshow(ky_exact_inner,cmap='bwr')
im2 = ax[1].imshow(wvs[1],cmap='bwr')
im3 = ax[2].imshow(np.abs(ky_exact_inner-wvs[1]),cmap='bwr')
ax[0].title.set_text('exact ky')
ax[1].title.set_text('GD Approximation')
ax[2].title.set_text('Abs Err')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/local_spec/"+"grad_descent_ky_compare_mu_{}.png".format(mu))
plt.close()


fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(4,14))
im1 = ax[0].imshow(wave_nums_exact_inner,cmap='bwr')
im2 = ax[1].imshow(wn_gd_approx,cmap='bwr')
im3 = ax[2].imshow(np.abs(wave_nums_exact_inner-wn_gd_approx),cmap='bwr')
ax[0].title.set_text('exact wave nums')
ax[1].title.set_text('GD Approximation')
ax[2].title.set_text('Abs Err')
plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])
plt.tight_layout()
plt.savefig(os.getcwd()+"/figs/local_spec/"+"grad_descent_wavenum_compare_mu_{}.png".format(mu))
plt.close()

logfile.close()