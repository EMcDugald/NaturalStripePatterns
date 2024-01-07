import numpy as np


def bilinear_interp(X, Y, W, xvals, yvals):
    if np.size(xvals) == 1:
        x_indices = np.abs(xvals - X[0, :]).argmin()
        y_indices = np.abs(yvals - Y[:, 0]).argmin()
    else:
        x_indices = np.abs(xvals[:, None] - X[0,:][None, :]).argmin(axis=-1)
        y_indices = np.abs(yvals[:, None] - Y[:,0][None, :]).argmin(axis=-1)
    x_mesh_vals = X[0,x_indices]
    y_mesh_vals = Y[y_indices,0]
    x1_flag = np.where((xvals-x_mesh_vals)<0,1,0)
    x1_indices = np.where((x1_flag)==1,x_indices-1,x_indices).astype(np.int64)
    x2_indices = np.where((x1_flag)==1,x_indices,x_indices+1).astype(np.int64)
    x1_vals = X[0,x1_indices]
    x2_vals = X[0,x2_indices]
    y1_flag = np.where((yvals-y_mesh_vals)<0,1,0)
    y1_indices = np.where((y1_flag)==1,y_indices-1,y_indices).astype(np.int64)
    y2_indices = np.where((y1_flag)==1,y_indices,y_indices+1).astype(np.int64)
    y1_vals = Y[y1_indices,0]
    y2_vals = Y[y2_indices,0]
    w11 = W[y1_indices,x1_indices]
    w21 = W[y1_indices,x2_indices]
    w12 = W[y2_indices,x1_indices]
    w22 = W[y2_indices,x2_indices]
    dx = x2_vals-x1_vals
    alpha1 = (x2_vals-xvals)/dx
    alpha2 = (xvals-x1_vals)/dx
    wxy1 = alpha1*w11 + alpha2*w21
    wxy2 = alpha1*w12 + alpha2*w22
    dy = (y2_vals-y1_vals)
    beta1 = (y2_vals-yvals)/dy
    beta2 = (yvals-y1_vals)/dy
    return beta1*wxy1 + beta2*wxy2


def snaking_indices(rows, cols, ss_fctr):
    indices = []
    for row in range(rows)[::ss_fctr]:
        if row % int(2*ss_fctr) == 0:  # Even rows
            for col in range(cols)[::ss_fctr]:
                indices.append((row, col))
        else:  # Odd rows
            for col in range(cols - ss_fctr, -1, -1)[::ss_fctr]:
                indices.append((row, col))
    return np.asarray(indices)