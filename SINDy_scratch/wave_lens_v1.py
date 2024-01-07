import numpy as np
from utils import bilinear_interp, snaking_indices
import scipy.io as sio
import os
import time
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


def kx(mu,x,y):
    k1 = np.sqrt(1 - mu ** 2)
    return k1

def ky(mu,x,y):
    k2 = mu
    return k2*np.tanh(k2*y)


def get_wave_lens_global(mu,max_wl,Lx,Ly,Nx,Ny,X,Y,W,phi_arr,wave_vec_aspect,wave_vec_subsample,fname='',save=True):
    start = time.time()
    sup_sample = 10
    max_L = min(.5*wave_vec_aspect[0]*Ly,.5*wave_vec_aspect[1]*Lx)
    L = max_L - .01*min(Lx,Ly)
    print("Sampling Length:",L)
    rect_y = int(wave_vec_aspect[0] * Ny)
    rect_x = int(wave_vec_aspect[1] * Nx)
    indices = snaking_indices(rect_y, rect_x, wave_vec_subsample)
    rect_y_ss = int(rect_y / wave_vec_subsample)
    rect_x_ss = int(rect_x / wave_vec_subsample)
    wave_len_arr = np.zeros(shape=(rect_y_ss, rect_x_ss))
    y_shift = int(.5 * (Ny - rect_y))
    x_shift = int(.5 * (Nx - rect_x))
    dl = np.sqrt((X[0, 1] - X[0, 0]) ** 2 + (Y[1, 0] - Y[0, 0]) ** 2)
    for pair in indices:
        r, c = pair
        print("sampling indices:", r + y_shift, c + x_shift)
        phi = phi_arr[int(r / wave_vec_subsample),int(c / wave_vec_subsample)]
        wl = get_wl(max_wl, X, Y, W, phi, r, y_shift, c, x_shift, L, dl, sup_sample)
        if wl == 0:
            wl = np.inf
            print("returned inf")
        print("wavelength:",wl)
        x = X[r + y_shift, c + x_shift]
        y = Y[r + y_shift, c + x_shift]
        exact_kx = kx(mu,x,y)
        exact_ky = ky(mu,x,y)
        exact_wn = np.sqrt(exact_kx**2 + exact_ky**2)
        approx_wn = 2*np.pi/wl
        err = np.abs(approx_wn-exact_wn)/exact_wn
        print("x0:",x,"y0:",y,"Approx vs. Exact Rel Err:",err)
        # if err > .1:
        #     n = round(10 * (L / dl))
        #     t = np.linspace(0, L, n)
        #     fwd_ws = get_fwd_ws(x, y, phi, X, Y, W, t)
        #     bwd_ws = get_bwd_ws(x, y, phi, X, Y, W, t)
        #     make_debug_plot(x, y, t, phi, X, Y, W, fwd_ws, bwd_ws, r + y_shift, c + x_shift,mu)
        wave_len_arr[int(r / wave_vec_subsample), int(c / wave_vec_subsample)] += wl
    if save:
        mdict = {'w': W, 'sampling_inds': indices,
                 'X': X, 'Y': Y,
                 'wave_lens': wave_len_arr}
        sio.savemat(os.getcwd() + "/data/wave_lens/" + fname + ".mat", mdict)
    end = time.time()
    print("Total time for wave lens:", end - start)
    return wave_len_arr


def get_wl(max_wl,X,Y,W,phi,r,y_shift,c,x_shift,L,dl,sup_sample):
    x0, y0, w0 = X[r + y_shift, c + x_shift], Y[r + y_shift, c + x_shift], W[r + y_shift, c + x_shift]
    n = round(sup_sample * (L / dl))
    t = np.linspace(0, L, n)
    fwd_ws = get_fwd_ws(x0,y0,phi,X,Y,W,t)
    bwd_ws = get_bwd_ws(x0,y0,phi,X,Y,W,t)
    #
    # if r + y_shift == 512 and c + x_shift == 1504:
    #     print("debug")

    is_bwd_ill_posed_flag = is_ill_posed_bwd(bwd_ws, fwd_ws, sup_sample)
    is_fwd_ill_posed_flag = is_ill_posed_fwd(bwd_ws, fwd_ws, sup_sample)

    if is_bwd_ill_posed_flag and is_fwd_ill_posed_flag:
        print("Both directions ill-posed")
        return np.inf

    elif is_fwd_ill_posed_flag and not is_bwd_ill_posed_flag:
        print("Moving only in backward direction")
        bwd_abs_diff = np.abs(bwd_ws - w0)
        # if ((w0 < np.mean(fwd_ws[1:int(sup_sample/2)])) and (w0 < np.mean(bwd_ws[1:int(sup_sample/2)]))) or \
        #         ((w0>np.mean(fwd_ws[1:int(sup_sample/2)])) and (w0>np.mean(bwd_ws[1:int(sup_sample/2)]))):
        if ((w0 < np.mean(fwd_ws[1:3])) and (w0 < np.mean(bwd_ws[1:3]))) or \
                ((w0 > np.mean(fwd_ws[1:3])) and (w0 > np.mean(bwd_ws[1:3]))):
            print("at a local extrema: only looking for one crossing")
            num_crossings = 1
            wl = get_crossing(bwd_abs_diff, t, num_crossings)
        else:
            print("not at a local extrema: looking for two crossings")
            num_crossings = 2
            wl = get_crossing(bwd_abs_diff, t, num_crossings)
        return wl

    elif not is_fwd_ill_posed_flag and is_bwd_ill_posed_flag:
        print("Moving only in forward direction")
        fwd_abs_diff = np.abs(fwd_ws - w0)
        # if ((w0 < np.mean(fwd_ws[1:int(sup_sample/2)])) and (w0 < np.mean(bwd_ws[1:int(sup_sample/2)]))) or \
        #         ((w0 > np.mean(fwd_ws[1:int(sup_sample/2)])) and (w0 > np.mean(bwd_ws[1:int(sup_sample/2)]))):
        if ((w0 < np.mean(bwd_ws[1:3])) and (w0 < np.mean(fwd_ws[1:3]))) or \
                ((w0 > np.mean(bwd_ws[1:3])) and (w0 > np.mean(fwd_ws[1:3]))):
            print("at a local extrema: only looking for one crossing")
            num_crossings = 1
            wl = get_crossing(fwd_abs_diff, t, num_crossings)
        else:
            print("not at a local extrema: looking for two crossings")
            num_crossings = 2
            wl = get_crossing(fwd_abs_diff, t, num_crossings)
        return wl

    elif ((w0<fwd_ws[1]) and (w0<bwd_ws[1])):
        print("at a local minimum")
        dist_to_bwd_max = get_next_max(bwd_ws,t,max_wl)
        dist_to_fwd_max = get_next_max(fwd_ws,t,max_wl)
        return dist_to_bwd_max + dist_to_fwd_max

    elif ((w0>fwd_ws[1]) and (w0>bwd_ws[1])):
        print("at a local maximum")
        dist_to_bwd_min = get_next_min(bwd_ws,t,max_wl)
        dist_to_fwd_min = get_next_min(fwd_ws,t,max_wl)
        return dist_to_bwd_min + dist_to_fwd_min

    elif ((w0 < fwd_ws[1]) and (w0 > bwd_ws[1])):
        print("not at a min or max; increasing in forward direction")
        dist_to_bwd_min = get_next_min(bwd_ws,t,max_wl)
        dist_to_fwd_max = get_next_max(fwd_ws,t,max_wl)
        return 2*(dist_to_bwd_min+dist_to_fwd_max)

    elif ((w0 > fwd_ws[1]) and (w0 < bwd_ws[1])):
        print("not at a min or max; decreasing in forward direction")
        dist_to_bwd_max = get_next_max(bwd_ws,t,max_wl)
        dist_to_fwd_min = get_next_min(fwd_ws,t,max_wl)
        return 2*(dist_to_bwd_max+dist_to_fwd_min)

    else:
        print("relative position of w0 undetermined")
        return np.inf



def get_fwd_ws(x0,y0,phi,X,Y,W,t):
    xs = x0 + t * np.cos(phi)
    ys = y0 + t * np.sin(phi)
    return bilinear_interp(X, Y, W, xs, ys)


def get_bwd_ws(x0,y0,phi,X,Y,W,t):
    xs = x0 + t * np.cos(phi+np.pi)
    ys = y0 + t * np.sin(phi+np.pi)
    return bilinear_interp(X, Y, W, xs, ys)


def get_next_max(profile,t,max_wl):
    wl = 0
    for j in range(1, len(profile) - 1):
        if (profile[j - 1] < profile[j]) and (profile[j + 1] < profile[j]):
            print("local max found")
            wl += t[j]
            break
    if wl == 0  or wl > max_wl:
        print("something went wrong")
        return np.inf
    return wl

def get_next_min(profile,t,max_wl):
    wl = 0
    for j in range(1, len(profile) - 1):
        if (profile[j - 1] > profile[j]) and (profile[j + 1] > profile[j]):
            print("local min found")
            wl += t[j]
            break
    if wl == 0  or wl > max_wl:
        print("something went wrong")
        return np.inf
    return wl

def make_debug_plot(x0,y0,t,phi,X,Y,W,fwd_ws,bwd_ws,row,col,mu):
    xs = x0 + t * np.cos(phi)
    ys = y0 + t * np.sin(phi)
    xstart = np.argmin(np.abs(X[0,:] - xs[0]))
    xend = np.argmin(np.abs(X[0,:] - xs[-1]))
    ystart = np.argmin(np.abs(Y[:,0] - ys[0]))
    yend = np.argmin(np.abs(Y[:,0] - ys[-1]))
    if xend < xstart:
        xstart = np.argmin(np.abs(X[0,:] - xs[-1]))
        xend = np.argmin(np.abs(X[0, :] - xs[0]))
    if yend < ystart:
        ystart = np.argmin(np.abs(Y[:, 0] - ys[-1]))
        yend = np.argmin(np.abs(Y[:, 0] - ys[0]))
    region = W[ystart-75:yend+75,xstart-75:xend+75]
    plt.scatter(X[ystart-75:yend+75,xstart-75:xend+75].flatten(),
                    Y[ystart-75:yend+75,xstart-75:xend+75].flatten(),
                    c=region.flatten())
    plt.scatter(xs,ys,c='r')
    plt.savefig(os.getcwd() + "/figs/debug/wavelens/" + "fwd_profile_{}_top_{}_{}.png".format(mu,row,col))
    plt.close()
    fig, ax = plt.subplots()
    ax.plot(t,fwd_ws)
    plt.savefig(os.getcwd() + "/figs/debug/wavelens/" + "fwd_profile_{}_sid_{}_{}.png".format(mu,row,col))
    plt.close()

    xs = x0 + t * np.cos(phi+np.pi)
    ys = y0 + t * np.sin(phi+np.pi)
    xstart = np.argmin(np.abs(X[0, :] - xs[0]))
    xend = np.argmin(np.abs(X[0, :] - xs[-1]))
    ystart = np.argmin(np.abs(Y[:, 0] - ys[0]))
    yend = np.argmin(np.abs(Y[:, 0] - ys[-1]))
    if xend < xstart:
        xstart = np.argmin(np.abs(X[0, :] - xs[-1]))
        xend = np.argmin(np.abs(X[0, :] - xs[0]))
    if yend < ystart:
        ystart = np.argmin(np.abs(Y[:, 0] - ys[-1]))
        yend = np.argmin(np.abs(Y[:, 0] - ys[0]))
    region = W[ystart - 75:yend + 75, xstart - 75:xend + 75]
    plt.scatter(X[ystart - 75:yend + 75, xstart - 75:xend + 75].flatten(),
                Y[ystart - 75:yend + 75, xstart - 75:xend + 75].flatten(),
                c=region.flatten())
    plt.scatter(xs, ys, c='r')
    plt.savefig(os.getcwd() + "/figs/debug/wavelens/" + "bwd_profile_{}_top_{}_{}.png".format(mu,row,col))
    plt.close()
    fig, ax = plt.subplots()
    ax.plot(t, bwd_ws)
    plt.savefig(os.getcwd() + "/figs/debug/wavelens/" + "bwd_profile_{}_sid_{}_{}.png".format(mu,row,col))
    plt.close()


def is_ill_posed_bwd(bwd_ws, fwd_ws,sup_sample):
    bwd_ws_smth = smooth(bwd_ws,sup_sample)
    fwd_ws_smth = smooth(fwd_ws,sup_sample)
    n_bwd_minima = len(argrelextrema(bwd_ws, np.less)[0])
    n_bwd_maxima = len(argrelextrema(bwd_ws, np.greater)[0])
    n_fwd_minima = len(argrelextrema(fwd_ws, np.less)[0])
    n_fwd_maxima = len(argrelextrema(fwd_ws, np.greater)[0])
    n_bwd_minima_smth = len(argrelextrema(bwd_ws_smth, np.less)[0])
    n_bwd_maxima_smth = len(argrelextrema(bwd_ws_smth, np.greater)[0])
    n_fwd_minima_smth = len(argrelextrema(fwd_ws_smth, np.less)[0])
    n_fwd_maxima_smth = len(argrelextrema(fwd_ws_smth, np.greater)[0])
    n_bwd_crit = n_bwd_minima + n_bwd_maxima
    n_fwd_crit = n_fwd_minima + n_fwd_maxima
    n_bwd_crit_smth = n_bwd_minima_smth + n_bwd_maxima_smth
    n_fwd_crit_smth = n_fwd_minima_smth + n_fwd_maxima_smth
    #if n_bwd_crit_smth < .5*n_bwd_crit: #this implies many oscillations
    if n_bwd_crit_smth <= .5 * n_bwd_crit:  # this implies many oscillations
        return True
    elif n_bwd_crit == 0: #this implies no critical points
        return True
    #elif np.abs(n_fwd_crit_smth-n_fwd_crit)<3 and n_fwd_crit / n_bwd_crit_smth > 2:
    elif np.abs(n_fwd_crit_smth - n_fwd_crit) < 3 and n_fwd_crit / n_bwd_crit_smth > 1.75:
        # this implies forward direction isn't ill posed and the backward direction has far fewer extrema
        return True
    else:
        return False


def is_ill_posed_fwd(bwd_ws, fwd_ws,sup_sample):
    bwd_ws_smth = smooth(bwd_ws, sup_sample)
    fwd_ws_smth = smooth(fwd_ws, sup_sample)
    n_bwd_minima = len(argrelextrema(bwd_ws, np.less)[0])
    n_bwd_maxima = len(argrelextrema(bwd_ws, np.greater)[0])
    n_fwd_minima = len(argrelextrema(fwd_ws, np.less)[0])
    n_fwd_maxima = len(argrelextrema(fwd_ws, np.greater)[0])
    n_bwd_minima_smth = len(argrelextrema(bwd_ws_smth, np.less)[0])
    n_bwd_maxima_smth = len(argrelextrema(bwd_ws_smth, np.greater)[0])
    n_fwd_minima_smth = len(argrelextrema(fwd_ws_smth, np.less)[0])
    n_fwd_maxima_smth = len(argrelextrema(fwd_ws_smth, np.greater)[0])
    n_bwd_crit = n_bwd_minima + n_bwd_maxima
    n_fwd_crit = n_fwd_minima + n_fwd_maxima
    n_bwd_crit_smth = n_bwd_minima_smth + n_bwd_maxima_smth
    n_fwd_crit_smth = n_fwd_minima_smth + n_fwd_maxima_smth
    #if n_fwd_crit_smth < .5*n_fwd_crit: #this implies many oscillations
    if n_fwd_crit_smth <= .5 * n_fwd_crit:  # this implies many oscillations
        return True
    elif n_fwd_crit == 0: #this implies no critical points
        return True
    #elif np.abs(n_bwd_crit_smth-n_bwd_crit)<3 and n_bwd_crit / n_fwd_crit_smth > 2:
    elif np.abs(n_bwd_crit_smth - n_bwd_crit) < 3 and n_bwd_crit / n_fwd_crit_smth >= 1.75:
        # this implies backward direction isn't ill posed and the forward direction has far fewer extrema
        return True
    else:
        return False


# def get_crossing(abs_diff,t,num_crossings):
#     ctr = 0
#     wl = 0
#     for j in range(1, len(abs_diff) - 1):
#         if (abs_diff[j - 1] > abs_diff[j]) and (abs_diff[j + 1] > abs_diff[j]):
#             ctr += 1
#             if ctr == num_crossings:
#                 wl += t[j]
#                 break
#     return wl


def get_crossing(abs_diff,t,num_crossings):
    ctr = 0
    wl = 0
    abs_diff_smooth = smooth(abs_diff,2)
    for j in range(1, len(abs_diff_smooth) - 1):
        if (abs_diff_smooth[j - 1] > abs_diff_smooth[j]) and (abs_diff_smooth[j + 1] > abs_diff_smooth[j]):
            ctr += 1
            if ctr == num_crossings:
                wl += t[j]
                break
    return wl


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth