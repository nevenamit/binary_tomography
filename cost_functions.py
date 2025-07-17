import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def _total_variation(X):
    N = int(np.sqrt(X.shape[0]))
    if N * N != X.shape[0]:
        raise ValueError("X should be a flattened square image")

    x_2d = X.reshape(N, N)

    gx = np.abs(np.diff(x_2d, axis=1)).sum()
    gy = np.abs(np.diff(x_2d, axis=0)).sum()

    return gx + gy

def _total_variation_2(X):
    N = int(np.sqrt(X.shape[0]))
    if N * N != X.shape[0]:
        raise ValueError("X should be a flattened square image")

    x_2d = X.reshape(N, N).astype(np.float32)
    kernel = np.array([[1, -1]])

    tv_x = np.abs(ndi.convolve(x_2d, kernel, mode='constant')).sum()
    tv_y = np.abs(ndi.convolve(x_2d, kernel.T, mode='constant')).sum()

    return tv_x + tv_y

def l2_norm(X, W, sino_target):
    return np.linalg.norm(W @ X - sino_target)

# --- Funkcija greške sa Totalnom Varijacijom (TV) regularizacijom ---
def l2_and_tv(X, W, sino_target, lambda_tv):

    l2 = l2_norm(X, W, sino_target)

    tv = _total_variation(X)

    return l2 + lambda_tv * tv


def _compute_spacial_lut(radius, sigma, print=False):
    x = np.arange(-radius, radius+1)
    X, Y = np.meshgrid(x, x)
    lut = np.exp(-(X**2 + Y**2)/(2*sigma**2))

    if print:
        fig =  plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, lut, cmap='coolwarm')

    return lut

def compute_regularization_term(
    image : np.ndarray,
    sigma : float, # variance of gaussian used
    r     : int, # radius of window used, size of window is 2r+1 x 2r+1
):
    spacial_lut = _compute_spacial_lut(r, sigma)

    N = image.shape[0]
    if image.shape[1] != image.shape[0]:
        raise ValueError("image should be square")
        
    cumulative_error = 0

    for i in range(N):
        for j in range(N):
            left_ind = max(j - r, 0)
            right_ind = min(j + r + 1, N)
            up_ind = max(i - r, 0)
            down_ind = min(i + r + 1, N)

            left_lut = r - (j - left_ind)
            right_lut = left_lut + (right_ind - left_ind)
            up_lut = r - (i - up_ind)
            down_lut = up_lut + (down_ind - up_ind) 

            slice = image[up_ind:down_ind, left_ind:right_ind]
            slice = np.abs(slice - image[i,j])

            lut = spacial_lut[up_lut:down_lut, left_lut:right_lut]
            
            cumulative_error += np.sum(slice * lut)
            
    return cumulative_error

def l2_gauss_regularization(
    X : np.ndarray,
    W : np.ndarray,
    sino_target : np.ndarray,
    sigma : float, # variance of gaussian used
    r     : int, # radius of window used, size of window is 2r+1 x 2r+1
):
    l2 = l2_norm(X, W, sino_target)
    gauss_reg = compute_regularization_term(X.reshape(int(np.sqrt(X.shape[0])), -1), sigma, r)

    return l2 + gauss_reg
