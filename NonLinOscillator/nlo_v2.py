import numpy as np
import matplotlib.pyplot as plt
import math


#integrate this to get x
def dxdt(y):
    return y


#integrate this to get x'
def dydt(x, y, eps):
    return -eps*x**3 - x + eps**2*y*(1-x**2)

#CASE 1 (theta0 = 0)
eps = 1/5
r0 = 1/2
x0 = 2*r0 + eps*r0**3/4
print("x0: ",x0, "x0^2:",x0**2, "x0^-1:",1/x0)
y0 = 0
###


tend = 4*np.pi/eps**2
t = np.linspace(0,tend,1000000)
dt = t[1]-t[0]
print("tend = ",tend)
print("dt = ",dt)

xvals = np.zeros_like(t)
yvals = np.zeros_like(t)
xvals[0] += x0
yvals[0] += y0

for i in range(1,len(t)):
    x = xvals[i-1]
    y = yvals[i-1]
    xvals[i] += x + dt*dxdt(y)
    yvals[i] += y + dt*dydt(x, y, eps)

fig, ax = plt.subplots(nrows=3,ncols=1)
ax[0].plot(t,xvals)
ax[0].set_xlim([t[0],t[-1]])
ax[1].plot(t,yvals)
ax[1].set_xlim([t[0],t[-1]])
ax[2].plot(xvals,yvals,c='k')
ax[0].set_title("x")
ax[1].set_title("x'")
ax[2].set_title("x, x' phase plane")
fig.suptitle("ODE Trajectories")
plt.tight_layout()
plt.show()

sum_sqs = (xvals**2 + yvals**2)
diff_sqs = (yvals**2 + xvals**2)

## averaging process ##
avg_len = np.pi/eps
conv_window = math.ceil(avg_len/dt)
print("num average points =",conv_window)
if conv_window%2 != 0:
    conv_window += 1
gap = int(conv_window/2)
avg_sum_sq = np.zeros(len(t)-conv_window)
for i in range(len(avg_sum_sq)):
    avg_sum_sq[i] += np.mean(sum_sqs[i:i+conv_window])

avg_diff_sq = np.zeros(len(t)-conv_window)
for i in range(len(avg_diff_sq)):
    avg_diff_sq[i] += np.mean(sum_sqs[i:i+conv_window])


rsq = -(1./8.)*avg_sum_sq + (1./4.)*avg_diff_sq

inner_t = t[gap:-gap]
fig, ax = plt.subplots(nrows=2,ncols=1)
ax[0].plot(inner_t,rsq)
ax[0].axvline(x=1/eps**2, c='r')
ax[0].set_xlim([inner_t[0],inner_t[-1]])
ax[1].plot(inner_t,sum_sqs[gap:-gap])
ax[1].axvline(x=1/eps**2, c='r')
ax[1].set_xlim([inner_t[0],inner_t[-1]])
ax[0].set_title('macro_rsq')
ax[1].set_title('sum of squares')
fig.suptitle('microscopic vs macroscopic rsq')
plt.tight_layout()
plt.show()

rsq_avg_theory = r0**2*np.exp((eps**2)*t)/(1-r0**2+r0**2*np.exp((eps**2)*t))
fig, ax = plt.subplots(nrows=2,ncols=1)
ax[0].plot(inner_t,rsq)
ax[0].set_xlim([inner_t[0],inner_t[-1]])
ax[1].plot(inner_t,rsq_avg_theory[gap:-gap])
ax[1].set_xlim([inner_t[0],inner_t[-1]])
ax[0].set_title('macro_rsq_estimate')
ax[1].set_title('macro_rsq_theoretical')
fig.suptitle('macro rsq estimate vs theoretical')
plt.tight_layout()
plt.show()

