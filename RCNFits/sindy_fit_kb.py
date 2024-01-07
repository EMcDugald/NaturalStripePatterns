import scipy.io as sio
import numpy as np
import os
from fit_pde import TrainSTRidge, print_pde
import sys

### This script perform the SINDy procedure on knee bend data- estimates of the wave vector and higher derivatives
### Note that J(k) and curl(k) are both zero (essentially). curl(k) is exactly 0, and is not a good candidate for the libraru
### Also note that theta_x is a constant in this case, so it is like putting 'one' in your library. thus, not a good candidate

#mu = .3
#mu = .5
mu = .7
#mu = .9
logfile = open(os.getcwd()+"/logs/knee_bends/sindy_for_kb_mu_{}.out".format(mu), 'w')
sys.stdout = logfile

data = sio.loadmat(os.getcwd()+"/data/knee_bends/mu_{}.mat".format(mu))
x = data['Xinterior'].flatten()
feature_vec_len = len(x)
theta_x = data['theta_x_approx'].flatten().reshape(feature_vec_len,1)
theta_y = data['theta_y_approx'].flatten().reshape(feature_vec_len,1)
divk = data['divk_approx'].flatten().reshape(feature_vec_len,1)
curlk = data['curlk_approx'].flatten().reshape(feature_vec_len,1)
Jk = data['Jk_approx'].flatten().reshape(feature_vec_len,1)
wavenum = data['wavenums_approx'].flatten().reshape(feature_vec_len,1)
one = np.ones_like(wavenum).reshape(feature_vec_len,1)

print("Self Dual Equation: (div(k))^2-1+2|k|^2-|k|^4 = 0")

print("First Fit LHS = div(k)^2")
print("RHS should be 1 - 2|k|^2 + |k|^4")
lib1 = np.hstack([wavenum, wavenum**2, wavenum**3,wavenum**4,
                 divk, divk**4, one, theta_y])
description1 = ['|k|', '|k|^2', '|k|^3', '|k|^4',
                 'div(k)', 'div(k)^4', '1','theta_y']
lhs1 = divk**2
c1 = TrainSTRidge(lib1,lhs1,1e-1,1e-12)
print("coefficient vector solution =",c1)
print_pde(c1, description1)

print("Second Fit LHS = div(k)^2 -1 + 2|k|^2 - |k|^4")
print("RHS should be 0")
lib2 = np.hstack([divk, divk**3, divk**4, wavenum,
                 wavenum**3, wavenum**5, np.log(wavenum),np.cos(theta_y)])
description2 = ['div(k)', 'div(k)^3', 'div(k)^4', '|k|',
                 '|k|^3', '|k|^5', 'ln(|k|)', 'cos(theta_y)']
lhs2 = divk**2 - 1 + 2*wavenum**2 - wavenum**4
c2 = TrainSTRidge(lib2,lhs2,1e-5,1e-8)
print("coefficient vector solution =",c2)
print_pde(c2, description2)


print("Third Fit LHS = 2|k|^2 - |k|^4")
print("RHS Should be: 1 - (div(k)^2")
lib3 = np.hstack([divk, divk**2, wavenum
                     , one, Jk, np.exp(divk), np.sin(theta_y)])
description3 = ['div(k)', 'div(k)^2', '|k|'
                , '1', 'Jk', 'e^(div(k))','sin(theta_y)']
lhs3 = 2*wavenum**2 - wavenum**4
c3 = TrainSTRidge(lib3,lhs3,1e-5,1e-8)
print("coefficient vector solution =",c3)
print_pde(c3, description3)



logfile.close()