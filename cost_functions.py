import numpy as np
import scipy.ndimage as ndi

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