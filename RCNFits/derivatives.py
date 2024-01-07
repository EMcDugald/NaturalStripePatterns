import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

def BackwardDiff(u_curr, u_past, dt):
    """
    :param u_curr: current time func values
    :param u_past: previous time func values
    :param dt: time step
    :return: backward finite diff approximation
    """
    return (u_curr - u_past)/dt


def FiniteDiffDerivs(func,dx,dy,type='x'):
    """
    :param func: 2d array of function values
    :param dx: grid spacing in "x" direction- variable that changes value along columns
    :param dy: grid spacing in "y" direction- variable that changes value along rows
    :param type: sets default derivative type
    :return: array of partial derivative values of specified type; based on centered differences

    Note: output shape will depend on order of derivative operator. order > 2 gets a shape of nx - 4, ny - 4
    """
    if type == 'x':
        #shape will be nx-2,ny-2
        d_arr = (func[1:-1,1:]-func[1:-1,:-1])[:,:-1]/dx
    elif type == 'xx':
        #shape will be nx-2,ny-2
        d_arr = (func[1:-1, :-2] - 2 * func[1:-1, 1:-1] + func[1:-1, 2:]) / dx ** 2
    elif type == 'y':
        # shape will be nx-2,ny-2
        d_arr = ((func[1:, 1:-1] - func[:-1, 1:-1]))[:-1,:]/dy
    elif type == 'yy':
        # shape will be nx-2,ny-2
        d_arr = (func[:-2, 1:-1] - 2 * func[1:-1, 1:-1] + func[2:, 1:-1]) / dy ** 2
    elif type == 'xy':
        # shape will be nx-4,ny-4
        fx = FiniteDiffDerivs(func,dx,dy,type='x')
        d_arr = FiniteDiffDerivs(fx,dx,dy,type='y')
    elif type == 'laplacian':
        # shape will be nx-2,ny-2
        fxx = FiniteDiffDerivs(func,dx,dy,type='xx')
        fyy = FiniteDiffDerivs(func,dx,dy,type='yy')
        d_arr = fxx+fyy
    elif type == 'biharmonic':
        # shape will be nx-4,ny-4
        lap = FiniteDiffDerivs(func,dx,dy,type='laplacian')
        d_arr = FiniteDiffDerivs(lap,dx,dy,type='laplacian')
    elif type == 'xxxx':
        # shape will be nx-4,ny-4
        fxx = FiniteDiffDerivs(func,dx,dy,type='xx')
        d_arr = FiniteDiffDerivs(fxx,dx,dy,type='xx')
    elif type == 'xxyy':
        # shape will be nx-4,ny-4
        fxx = FiniteDiffDerivs(func,dx,dy,type='xx')
        d_arr = FiniteDiffDerivs(fxx,dx,dy,type='yy')
    elif type == 'yyyy':
        # shape will be nx-4,ny-4
        fyy = FiniteDiffDerivs(func,dx,dy,type='yy')
        d_arr = FiniteDiffDerivs(fyy,dx,dy,type='yy')
    elif type == 'xxx':
        # shape will be nx-4,ny-4
        fxx = FiniteDiffDerivs(func,dx,dy,type='xx')
        d_arr = FiniteDiffDerivs(fxx,dx,dy,type='x')
    elif type == 'yyy':
        # shape will be nx-4,ny-4
        fyy = FiniteDiffDerivs(func,dx,dy,type='yy')
        d_arr = FiniteDiffDerivs(fyy,dx,dy,type='y')
    elif type == 'xxy':
        # shape will be nx-4,ny-4
        fxx = FiniteDiffDerivs(func,dx,dy,type='xx')
        d_arr = FiniteDiffDerivs(fxx,dx,dy,type='y')
    elif type == 'xyy':
        # shape will be nx-4,ny-4
        fyy = FiniteDiffDerivs(func,dx,dy,type='yy')
        d_arr = FiniteDiffDerivs(fyy,dx,dy,type='x')
    else:
        raise Exception("Incompatible type selection")
    return d_arr

def SpectralDerivs(func,Lx,Ly,type='x'):
    """
    :param func: function values
    :param Lx: length of rectangle in x dir (var changing along columns)
    :param Ly: length of rectangle in y dir (var changing along rows)
    :param type: derivative type: 'x', 'xx',
    'y', 'yy', 'xy', 'xxyy', 'xxxx', 'yyyy',
    'laplacian', 'biharmonic'
    :return: grid of derivative values
    """
    #ny, nx = np.shape(func)
    nx, ny = np.shape(func)
    kx = (2.*np.pi/Lx)*fftfreq(nx,1./nx)
    ky = (2.*np.pi/Ly)*fftfreq(ny,1./ny)
    Kx, Ky = np.meshgrid(kx, ky)
    #Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    if type == 'x':
        return np.real(ifft2(1j*Kx*fft2(func)))
    elif type == 'xx':
        return np.real(ifft2((1j*Kx)**2*fft2(func)))
    elif type == 'y':
        return np.real(ifft2(1j*Ky*fft2(func)))
    elif type == 'yy':
        return np.real(ifft2((1j*Ky)**2*fft2(func)))
    elif type == 'xy':
        return np.real(ifft2((1j*Ky)*(1j*Kx)*fft2(func)))
    elif type == 'xxyy':
        return np.real(ifft2((1j*Ky)**2*(1j*Kx)**2*fft2(func)))
    elif type == 'xxxx':
        return np.real(ifft2((1j*Kx)**4*fft2(func)))
    elif type == 'yyyy':
        return np.real(ifft2((1j*Ky)**4*fft2(func)))
    elif type == 'laplacian':
        fourierLaplacian = -(Kx**2+Ky**2)
        return np.real(ifft2(fourierLaplacian*fft2(func)))
    elif type == 'biharmonic':
        fourierLaplacian = -(Kx**2+Ky**2)
        fourierBiharm = fourierLaplacian*fourierLaplacian
        return np.real(ifft2(fourierBiharm*fft2(func)))
    else:
        raise Exception("Incompatible type selection")