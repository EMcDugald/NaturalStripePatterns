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
y0 = 0
###


tend = 4*np.pi/eps**2
t = np.linspace(0,tend,100000)
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

rsq = xvals**2 + yvals**2

## averaging process ##
# avg_len = eps
# conv_window = math.ceil(tend/avg_len)
avg_len = np.pi/eps
# avg_len = np.pi/eps**2
conv_window = math.ceil(avg_len/dt)
print("num average points =",conv_window)
if conv_window%2 != 0:
    conv_window += 1
gap = int(conv_window/2)
rsq_avg = np.zeros(len(t)-conv_window)
for i in range(len(rsq_avg)):
    rsq_avg[i] += np.mean(rsq[i:i+conv_window])

inner_t = t[gap:-gap]
inner_rsq = rsq[gap:-gap]
fig, ax = plt.subplots(nrows=2,ncols=1)
ax[0].plot(inner_t,inner_rsq)
ax[0].axvline(x=1/eps**2, c='r')
ax[0].set_xlim([inner_t[0],inner_t[-1]])
ax[1].plot(inner_t,rsq_avg)
ax[1].axvline(x=1/eps**2, c='r')
ax[1].set_xlim([inner_t[0],inner_t[-1]])
ax[0].set_title('micro_rsq')
ax[1].set_title('macro_rsq')
fig.suptitle('microscopic vs macroscopic rsq')
plt.tight_layout()
plt.show()

rsq_avg_theory = r0**2*np.exp(eps**2*t)/(1-r0**2+r0**2*np.exp(eps**2*t))
fig, ax = plt.subplots(nrows=2,ncols=1)
ax[0].plot(inner_t,rsq_avg)
ax[0].set_xlim([inner_t[0],inner_t[-1]])
ax[1].plot(inner_t,rsq_avg_theory[gap:-gap])
ax[1].set_xlim([inner_t[0],inner_t[-1]])
ax[0].set_title('macro_rsq_estimate')
ax[1].set_title('macro_rsq_theoretical')
fig.suptitle('macro rsq estimate vs theoretical')
plt.tight_layout()
plt.show()

davgr_dt = (rsq_avg[2:]-rsq_avg[:-2])/(2*dt)
davgr_dt_theory = eps**2*rsq_avg*(1-rsq_avg)
davgr_dt_theory = davgr_dt_theory[1:-1]
fig, ax = plt.subplots(nrows=2,ncols=1)
ax[0].plot(inner_t[1:-1],davgr_dt)
ax[0].set_xlim([inner_t[1:-1][0],inner_t[1:-1][-1]])
ax[1].plot(inner_t[1:-1],davgr_dt_theory)
ax[1].set_xlim([inner_t[1:-1][0],inner_t[1:-1][-1]])
ax[0].set_title('avg_rsq_deriv_est')
ax[1].set_title('avg_rsq_deriv_theory')
fig.suptitle('macro rsq derivative estimate vs theoretical')
plt.tight_layout()
plt.show()

### average the average ###
avg_len = np.pi/eps
# avg_len = np.pi/eps**2
conv_window = math.ceil(avg_len/dt)
print("num average points =",conv_window)
if conv_window%2 != 0:
    conv_window += 1
gap = int(conv_window/2)
rsq_avg_avg = np.zeros(len(inner_t)-conv_window)
for i in range(len(rsq_avg_avg)):
    rsq_avg_avg[i] += np.mean(rsq_avg[i:i+conv_window])

inner_inner_t = inner_t[gap:-gap]
inner_inner_rsq = inner_rsq[gap:-gap]
fig, ax = plt.subplots(nrows=2,ncols=1)
ax[0].plot(inner_inner_t,inner_inner_rsq)
ax[0].axvline(x=2*np.pi/eps**2, c='r')
ax[0].set_xlim([inner_inner_t[0],inner_inner_t[-1]])
ax[1].plot(inner_inner_t,rsq_avg_avg)
ax[1].axvline(x=2*np.pi/eps**2, c='r')
ax[1].set_xlim([inner_inner_t[0],inner_inner_t[-1]])
ax[0].set_title('micro_rsq')
ax[1].set_title('macro_rsq')
fig.suptitle('microscopic vs macroscopic rsq with more averaging')
plt.tight_layout()
plt.show()

rsq_avg_theory = r0**2*np.exp(eps**2*t)/(1-r0**2+r0**2*np.exp(eps**2*t))
fig, ax = plt.subplots(nrows=2,ncols=1)
ax[0].plot(inner_inner_t,rsq_avg_avg)
ax[0].set_xlim([inner_inner_t[0],inner_inner_t[-1]])
ax[1].plot(inner_inner_t,rsq_avg_theory[gap:-gap][gap:-gap])
ax[1].set_xlim([inner_inner_t[0],inner_inner_t[-1]])
ax[0].set_title('macro_rsq_estimate')
ax[1].set_title('macro_rsq_theoretical')
fig.suptitle('macro rsq estimate vs theoretical with more averaging')
plt.tight_layout()
plt.show()


davgr_dt = (rsq_avg_avg[2:]-rsq_avg_avg[:-2])/(2*dt)
davgr_dt_theory = eps**2*rsq_avg_avg*(1-rsq_avg_avg)
davgr_dt_theory = davgr_dt_theory[1:-1]
fig, ax = plt.subplots(nrows=2,ncols=1)
ax[0].plot(inner_inner_t[1:-1],davgr_dt)
ax[0].set_xlim([inner_inner_t[1:-1][0],inner_inner_t[1:-1][-1]])
ax[1].plot(inner_inner_t[1:-1],davgr_dt_theory)
ax[1].set_xlim([inner_inner_t[1:-1][0],inner_inner_t[1:-1][-1]])
ax[0].set_title('avg_rsq_deriv_est')
ax[1].set_title('avg_rsq_deriv_theory')
fig.suptitle('macro rsq derivative estimate vs theoretical with more averaging')
plt.tight_layout()
plt.show()
