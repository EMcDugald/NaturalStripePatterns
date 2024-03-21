import scipy.io as sio
import numpy as np
import os
from fit_pde import TrainSTRidge, print_pde
import sys

### This script perform the SINDy procedure on RCN dislocation
### Note that J(k) and curl(k) are both zero (essentially). curl(k) is exactly 0, and is not a good candidate for the libraru


logfile = open(os.getcwd()+"/logs/dislocation/sindy_fit_dislocation_v5.out", 'w')
sys.stdout = logfile

data = sio.loadmat(os.getcwd()+"/data/dislocation/v5.mat")
feature_vec_len = len(data['theta_x_approx'].flatten()[~np.isnan(data['theta_x_approx'].flatten())])
theta_x = data['theta_x_approx'].flatten()[~np.isnan(data['theta_x_approx'].flatten())].reshape(feature_vec_len,1)
theta_y = data['theta_y_approx'].flatten()[~np.isnan(data['theta_y_approx'].flatten())].reshape(feature_vec_len,1)
divk = data['divk_approx'].flatten()[~np.isnan(data['divk_approx'].flatten())].reshape(feature_vec_len,1)
curlk = data['curlk_approx'].flatten()[~np.isnan(data['curlk_approx'].flatten())].reshape(feature_vec_len,1)
Jk = data['Jk_approx'].flatten()[~np.isnan(data['Jk_approx'].flatten())].reshape(feature_vec_len,1)
wavenum = data['wavenums_approx'].flatten()[~np.isnan(data['wavenums_approx'].flatten())].reshape(feature_vec_len,1)
Fx = data['F_x_approx'].flatten()[~np.isnan(data['F_x_approx'].flatten())].reshape(feature_vec_len,1)
Fyy = data['F_yy_approx'].flatten()[~np.isnan(data['F_yy_approx'].flatten())].reshape(feature_vec_len,1)
F = data['F_approx'].flatten()[~np.isnan(data['F_approx'].flatten())].reshape(feature_vec_len,1)
one = np.ones_like(wavenum).reshape(len(wavenum),1)

print("Self Dual Equation: (div(k))^2-1+2|k|^2-|k|^4 = 0")
print("Actual SD Balance (L2):",np.linalg.norm(divk**2-one+2*wavenum**2-wavenum**4))
print("Actual SD Balance (Mean AE):",np.mean(np.abs(divk**2-one+2*wavenum**2-wavenum**4)))
print("Actual SD Balance (Max AE):",np.max(np.abs(divk**2-one+2*wavenum**2-wavenum**4)))

print("First Fit LHS = div(k)^2")
print("RHS should be 1 - 2|k|^2 + |k|^4")
lib1 = np.hstack([wavenum**2,wavenum**4,one])
description1 = ['|k|^2', '|k|^4', '1']
lhs1 = divk**2
#c1 = TrainSTRidge(lib1,lhs1,1e-1,1e-12,maxit=25, STR_iters=10,l0_penalty=.1)
c1 = TrainSTRidge(lib1,lhs1,1e-1,1e-5,maxit=25, STR_iters=10)
print("coefficient vector solution =",c1)
print_pde(c1, description1)
print("Error 1:", np.linalg.norm(lhs1 - (1-2*wavenum**2+wavenum**4)))

print("Second Fit LHS = div(k)^2 -1 + 2|k|^2 - |k|^4")
print("RHS should be 0")
lib2 = np.hstack([divk, wavenum,
                 wavenum**3])
description2 = ['div(k)', '|k|',
                 '|k|^3']
lhs2 = divk**2 - 1 + 2*wavenum**2 - wavenum**4
c2 = TrainSTRidge(lib2,lhs2,1e-5,1e-1,maxit=25, STR_iters=10)
print("coefficient vector solution =",c2)
print_pde(c2, description2)
print("Error 2:", np.linalg.norm(lhs2 -0))


print("Third Fit LHS = 2|k|^2 - |k|^4")
print("RHS Should be: 1 - (div(k)^2")
lib3 = np.hstack([divk**2, wavenum
                     ,one, Jk])

description3 = ['div(k)^2', '|k|'
                , '1', 'Jk']
lhs3 = 2*wavenum**2 - wavenum**4
c3 = TrainSTRidge(lib3,lhs3,1e-1,1e-16,maxit=25, STR_iters=10)
print("coefficient vector solution =",c3)
print_pde(c3, description3)
print("Error 3:", np.linalg.norm(lhs3 - (1-divk**2)))


kb = data['kb'][0][0]
beta = data['beta'][0][0]
print("F Equation: 2 beta kb F_x + F_yy = 0")
print("Actual F Balance (L2):",np.linalg.norm(2*kb*beta*Fx+Fyy))
print("Actual F Balance (Mean AE):",np.mean(np.abs(2*kb*beta*Fx+Fyy)))
print("Actual F Balance (Max AE):",np.max(np.abs(2*kb*beta*Fx+Fyy)))

print("F Equation Fits")
print("F Fit LHS = 2*kbeta*kb*Fx+Fyy", "beta = ", beta, "kb = ", kb)
print("RHS should be 0")
lib1 = np.hstack([F, F**2, wavenum])
description1 = ['F','F^2','|k|']
lhs1 = 2*beta*kb*Fx+Fyy
c1 = TrainSTRidge(lib1,lhs1,10**-5,1,maxit=25, STR_iters=10)
print("coefficient vector solution =",c1)
print_pde(c1, description1)

print("F Fit LHS = 2*kbeta*kb*Fx", "beta = ", beta, "kb = ", kb)
print("RHS should be -Fyy")
lib2 = np.hstack([Fyy, F**2, wavenum, Fyy**2])
description2 = ['Fyy', 'F^2', '|k|', 'Fyy^2']
lhs2 = 2*beta*kb*Fx
c2 = TrainSTRidge(lib2,lhs2,10**-5,1,maxit=25, STR_iters=10)
print("coefficient vector solution =",c2)
print_pde(c2, description2)


logfile.close()