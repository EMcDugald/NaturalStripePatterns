import numpy as np
from utils import bilinear_interp, snaking_indices
from derivatives import FiniteDiffDerivs
import scipy.io as sio
import os
import time
import matplotlib.pyplot as plt


def get_wave_dirs(mu,Nx,Ny,X,Y,W,wave_vec_aspect,wave_vec_subsample,fname='',save=True):
    exact_phis = get_exact_wave_dirs(mu,X,Y,Nx,Ny,wave_vec_aspect,wave_vec_subsample)
    start = time.time()
    rect_y = int(wave_vec_aspect[0]*Ny)
    rect_x = int(wave_vec_aspect[1]*Nx)
    indices = snaking_indices(rect_y, rect_x, wave_vec_subsample)
    rect_y_ss = int(rect_y/wave_vec_subsample)
    rect_x_ss = int(rect_x/wave_vec_subsample)
    phi_arr = np.zeros(shape=(rect_y_ss, rect_x_ss))
    y_shift = int(.5*(Ny-rect_y))
    x_shift = int(.5*(Nx-rect_x))
    dl = np.sqrt((X[0,1]-X[0,0])**2+(Y[1,0]-Y[0,0])**2)
    print("characteristic length:",dl)
    for pair in indices:
        r, c = pair
        x0 = X[r + y_shift, c + x_shift]
        y0 = Y[r + y_shift, c + x_shift]
        eps = .25 * dl
        dist = min(1.,np.abs(y0))
        if dist < 1:
            #eps = min(1.5*dl,eps/dist)
            eps = max(.01*dl,dist*eps)
        w0 = W[r + y_shift, c + x_shift]
        print("disk radius:",eps)
        print("sampling indices:", r + y_shift, c + x_shift)
        num_samples = min(2500,round(500/dist))
        phi = get_phi(eps,x0,y0,w0,X,Y,W,num_samples)
        exact_phi = exact_phis[int(r/wave_vec_subsample),int(c/wave_vec_subsample)]
        print("x0:",x0,"y0:",y0,"Approx Phi:",phi,"Exact Phi:",exact_phi, "num_samples:",num_samples)
        cos_err = np.abs(np.abs(np.cos(phi))-np.abs(np.cos(exact_phi)))/np.abs(np.cos(exact_phi))
        sin_err = np.abs(np.abs(np.sin(phi))-np.abs(np.sin(exact_phi)))/np.abs(np.sin(exact_phi))
        print("|cos(phi)| rel err:",cos_err)
        print("|sin(phi)| rel err:",sin_err)
        # if sin_err > .3:
        #    make_debug_plot(x0, y0, w0, phi, X, Y, W, r + y_shift, c + x_shift, eps, exact_phi, num_samples, mu)
        phi_arr[int(r / wave_vec_subsample), int(c / wave_vec_subsample)] += phi
    if save:
        mdict = {'w': W,'sampling_inds': indices,
                 'X': X, 'Y': Y,
                 'thetas': phi_arr}
        sio.savemat(os.getcwd()+"/data/wave_dirs/"+fname+".mat",mdict)
    end = time.time()
    print("Total time for wave directions:",end-start)
    return phi_arr


def get_phi(eps,x0,y0,w0,X,Y,W,n):
    """
    Makes three discs around x0,y0 of size .8eps, eps, 1.2eps, and averages the direction of max difference
    """
    t = np.linspace(0,2*np.pi,n)
    xvals = x0 + eps*np.cos(t)
    yvals = y0 + eps*np.sin(t)
    ws = bilinear_interp(X,Y,W,xvals,yvals)
    delta_ws = np.abs(ws-w0)
    return t[np.argmax(delta_ws)]


def get_phi_from_derivs(Nx,Ny,wave_vec_aspect,wave_vec_subsample,X,Y,W,saveim=False):
    phi = np.zeros((Ny,Nx))
    dx = X[0,1]-X[0,0]
    dy = Y[1,0]-Y[0,0]
    dwdx = FiniteDiffDerivs(W,dx,dy,type='x')
    dwdy = FiniteDiffDerivs(W,dx,dy,type='y')
    phi[1:-1,1:-1] += np.arctan2(dwdy,dwdx)
    if saveim:
        fig, ax = plt.subplots()
        ax.imshow(dwdx,cmap='bwr')
        plt.savefig(os.getcwd()+"/figs/debug/"+"fd_derivs_x.png")
        fig, ax = plt.subplots()
        ax.imshow(dwdy,cmap='bwr')
        plt.savefig(os.getcwd()+"/figs/debug/" + "fd_derivs_y.png")
        fig, ax = plt.subplots()
        ax.imshow(phi, cmap='bwr')
        plt.savefig(os.getcwd() + "/figs/debug/" + "phi_from_fd.png")
    rect_y = int(wave_vec_aspect[0] * Ny)
    rect_x = int(wave_vec_aspect[1] * Nx)
    indices = snaking_indices(rect_y, rect_x, wave_vec_subsample)
    rect_y_ss = int(rect_y / wave_vec_subsample)
    rect_x_ss = int(rect_x / wave_vec_subsample)
    phi_arr = np.zeros(shape=(rect_y_ss, rect_x_ss))
    y_shift = int(.5*(Ny - rect_y))
    x_shift = int(.5*(Nx - rect_x))
    for pair in indices:
        r, c = pair
        phi_arr[int(r / wave_vec_subsample), int(c / wave_vec_subsample)] += phi[r+y_shift,c+x_shift]
    return phi_arr

def kx(x,y,mu):
    k1 = np.sqrt(1 - mu ** 2)
    return k1*np.ones(shape=np.shape(x))

def ky(x,y,mu):
    k2 = mu
    return k2*np.tanh(k2*y)

def get_exact_wave_dirs(mu,X,Y,Nx,Ny,wave_vec_aspect,wave_vec_subsample):
    kx_exact = kx(X, Y, mu)
    ky_exact = ky(X, Y, mu)
    phi_kexact = np.arctan2(ky_exact, kx_exact)
    rect_y = int(wave_vec_aspect[0] * Ny)
    rect_x = int(wave_vec_aspect[1] * Nx)
    indices = snaking_indices(rect_y, rect_x, wave_vec_subsample)
    rect_y_ss = int(rect_y / wave_vec_subsample)
    rect_x_ss = int(rect_x / wave_vec_subsample)
    wd_kexact = np.zeros(shape=(rect_y_ss, rect_x_ss))
    y_shift = int(.5 * (Ny - rect_y))
    x_shift = int(.5 * (Nx - rect_x))
    for pair in indices:
        r, c = pair
        wd_kexact[int(r / wave_vec_subsample), int(c / wave_vec_subsample)] += phi_kexact[r + y_shift, c + x_shift]
    return wd_kexact


def make_debug_plot(x0, y0, w0, phi, X, Y, W, row,col, eps, exact_phi,num_samples,mu):
    t = np.linspace(0, 2 * np.pi, num_samples)
    xvals = x0 + eps * np.cos(t)
    yvals = y0 + eps * np.sin(t)
    xmin = np.min(xvals)
    xmax = np.max(xvals)
    xrng = xmax - xmin
    ymin = np.min(yvals)
    ymax = np.max(yvals)
    yrng = ymax-ymin
    xstart = xmin - .25*xrng
    xend = xmax + .25*xrng
    ystart = ymin - .25*yrng
    yend = ymax + .25*yrng
    xs = np.linspace(xstart,xend,100)
    ys = np.linspace(ystart,yend,100)
    Xs, Ys = np.meshgrid(xs,ys)
    region = bilinear_interp(X,Y,W,Xs.flatten(),Ys.flatten())
    fig, ax = plt.subplots()
    ax.scatter(Xs.flatten(),Ys.flatten(),c=region.flatten())
    ax.plot(xvals,yvals,c='k')
    ax.scatter(x0,y0,c='g')
    ax.scatter(x0+eps*np.cos(phi),y0+eps*np.sin(phi),c='r')
    ax.scatter(x0 + eps * np.cos(exact_phi), y0 + eps * np.sin(exact_phi), c='k')
    ax.scatter(x0 - eps * np.cos(phi), y0 - eps * np.sin(phi), c='r')
    im = ax.scatter(x0 - eps * np.cos(exact_phi), y0 - eps * np.sin(exact_phi), c='k')
    fig.colorbar(im,ax=ax)
    plt.savefig(os.getcwd() + "/figs/debug/directions/" + "disk_on_plane_{}_{}_{}.png".format(mu,row,col))
    plt.close()

    ws = bilinear_interp(X, Y, W, xvals, yvals)
    delta_ws = np.abs(ws - w0)
    fig, ax = plt.subplots()
    ax.scatter(xvals, yvals, c=delta_ws)
    ax.scatter(x0, y0, c='g')
    ax.scatter(x0 + eps * np.cos(phi), y0 + eps * np.sin(phi), c='r')
    ax.scatter(x0 + eps * np.cos(exact_phi), y0 + eps * np.sin(exact_phi), c='k')
    ax.scatter(x0 - eps * np.cos(phi), y0 - eps * np.sin(phi), c='r')
    im = ax.scatter(x0 - eps * np.cos(exact_phi), y0 - eps * np.sin(exact_phi), c='k')
    fig.colorbar(im, ax=ax)
    plt.savefig(os.getcwd() + "/figs/debug/directions/" + "diffs_on_disk_{}_{}_{}.png".format(mu,row, col))
    plt.close()