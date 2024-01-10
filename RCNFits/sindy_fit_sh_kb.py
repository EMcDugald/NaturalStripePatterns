import scipy.io as sio
import numpy as np
import os
from fit_pde import TrainSTRidge, print_pde
import sys

### This script perform the SINDy procedure on knee bend data- estimates of the wave vector and higher derivatives
### Note that J(k) and curl(k) are both zero (essentially). curl(k) is exactly 0, and is not a good candidate for the libraru
### Also note that theta_x is a constant in this case, so it is like putting 'one' in your library. thus, not a good candidate


logfile = open(os.getcwd()+"/logs/sh_pgbs/sindy_fit_sh_kb_mu_0.3.out", 'w')
sys.stdout = logfile

data = sio.loadmat(os.getcwd()+"/data/sh_pgbs/v5_mu_0.3.mat")
x = data['Xinterior'].flatten()
feature_vec_len = len(x)
theta_x = data['theta_x_approx'].flatten().reshape(feature_vec_len,1)[::2]
theta_y = data['theta_y_approx'].flatten().reshape(feature_vec_len,1)[::2]
divk = data['divk_approx'].flatten().reshape(feature_vec_len,1)[::2]
curlk = data['curlk_approx'].flatten().reshape(feature_vec_len,1)[::2]
Jk = data['Jk_approx'].flatten().reshape(feature_vec_len,1)[::2]
wavenum = data['wavenums_approx'].flatten().reshape(feature_vec_len,1)[::2]
one = np.ones_like(wavenum).reshape(len(wavenum),1)

print("Self Dual Equation: (div(k))^2-1+2|k|^2-|k|^4 = 0")

print("First Fit LHS = div(k)^2")
print("RHS should be 1 - 2|k|^2 + |k|^4")
lib1 = np.hstack([wavenum, wavenum**2,wavenum**4,
                 divk, divk**4, one, theta_y])
description1 = ['|k|', '|k|^2', '|k|^4',
                 'div(k)', 'div(k)^4', '1','theta_y']
lhs1 = divk**2
c1 = TrainSTRidge(lib1,lhs1,1e-1,1e-12,maxit=25, STR_iters=10,l0_penalty=.1)
print("coefficient vector solution =",c1)
print_pde(c1, description1)

print("Second Fit LHS = div(k)^2 -1 + 2|k|^2 - |k|^4")
print("RHS should be 0")
lib2 = np.hstack([divk, divk**3, divk**4, wavenum,
                 wavenum**3, wavenum**5, np.log(wavenum)])
description2 = ['div(k)', 'div(k)^3', 'div(k)^4', '|k|',
                 '|k|^3', '|k|^5', 'ln(|k|)']
lhs2 = divk**2 - 1 + 2*wavenum**2 - wavenum**4
c2 = TrainSTRidge(lib2,lhs2,1e-4,1e-4,maxit=25, STR_iters=10)
print("coefficient vector solution =",c2)
print_pde(c2, description2)


print("Third Fit LHS = 2|k|^2 - |k|^4")
print("RHS Should be: 1 - (div(k)^2")
lib3 = np.hstack([divk, divk**2, wavenum
                     , one, Jk, np.exp(divk), np.sin(theta_y)])
description3 = ['div(k)', 'div(k)^2', '|k|'
                , '1', 'Jk', 'e^(div(k))','sin(theta_y)']
lhs3 = 2*wavenum**2 - wavenum**4
c3 = TrainSTRidge(lib3,lhs3,1e-1,1e-16,maxit=25, STR_iters=10)
print("coefficient vector solution =",c3)
print_pde(c3, description3)



logfile.close()