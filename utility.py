import numpy as np
import matplotlib.pyplot as plt
import skimage as ski


def plot_images(images  : list,
                titles  : list,
                rows    : int = None,
                cols    : int = None
                ) -> None:
    """
    Plot a list of images with their respective titles.

    This is a function for plotting multiple images with some 
    predetermined parameters, helps to avoid repeating code.

    Parameters
    ----------
    images : list
        A list of images to be plotted.
    
    titles : list
        A list of titles for each image.

    rows : int, optional
        Number of rows in the plot. If not provided, it will be calculated
        as ceil(n_images / cols). Default is None.
    
    cols : int, optional
        Number of columns in the plot. If not provided, it will be calculated
        as ceil(n_images / rows). Default is None.
    
    Returns
    -------
    None
    """

    assert type(images) == list, 'images must be a list'
    assert type(titles) == list, 'titles must be a list'
    assert len(images) == len(titles), 'images and titles must have the same length'
    if rows is not None and cols is not None:
        assert rows*cols >= len(images), 'rows*cols must be greater or equal to the number of images'

    n = len(images)

    ratio = images[0].shape[1] / images[0].shape[0]
    width = 6
    height = width / ratio

    if n == 1:
        fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=300)
        if images[0].ndim==2:
            ax.imshow(images[0], cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(images[0])
        ax.set_title(titles[0])
        ax.axis('off')
        return
    
    if rows is None and cols is None:
        cols = 1 if n == 1 else 2
        rows = np.ceil(n / cols).astype(int)
    elif rows is None:
        rows = np.ceil(n / cols).astype(int)
    elif cols is None:
        cols = np.ceil(n / rows).astype(int)

    fig, ax = plt.subplots(rows, cols, figsize=(width*cols, height*rows), dpi=120)
    fig.tight_layout()
    ax = ax.ravel()

    for i in range(n):
        if images[i].ndim == 2:
            if np.max(images[i]) > 1:
                images[i] = ski.img_as_float(images[i])
            ax[i].imshow(images[i], cmap='gray')
        else:
            ax[i].imshow(images[i])
        ax[i].set_title(titles[i])
        ax[i].axis('off') 


def cnotch(filt_type, notch, Nx, Ny, C, r, n=1):

    N_filters = len(C)
    
    filter_mask = np.zeros([Nx, Ny])
    
    if (Ny%2 == 0):
        y = np.arange(0,Ny) - Ny/2 + 0.5
    else:
        y = np.arange(0,Ny) - (Ny-1)/2
    
    if (Nx%2 == 0):
        x = np.arange(0,Nx) - Nx/2 + 0.5
    else:
        x = np.arange(0,Nx) - (Nx-1)/2

    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for i in range(0, N_filters):
        C_current = C[i]
        
        if (Ny%2 == 0):
            y0 = y - C_current[1] + Ny/2 - 0.5
        else:
            y0 = y - C_current[1] + (Ny-1)/2
        
        if (Nx%2 == 0):
            x0 = x - C_current[0] + Nx/2 - 0.5
        else:
            x0 = x - C_current[0] + (Nx-1)/2
        
        X0, Y0 = np.meshgrid(x0, y0, indexing='ij')
        
        # D0 = np.sqrt(np.square(X0) + np.square(Y0))
        D0 = np.hypot(X0, Y0)
    
        if filt_type == 'gaussian':
            filter_mask = filter_mask + \
                          np.exp(-np.square(D0)/(2*np.square(r)))

        elif filt_type == 'btw':
            filter_mask = filter_mask + \
                          1/(1+(D0/r)**(2*n))

        elif filt_type == 'ideal':
            filter_mask[D0<=r] = 1

        else:
            print('Greška! Nije podržan tip filtra: ', filt_type)
            return
        
    filter_mask = filter_mask / np.max(filter_mask)

    if notch == 'pass':
        return filter_mask
    elif notch == 'reject':
        return 1 - filter_mask
    else:
        return


def lpfilter(filt_type, Nx, Ny, sigma, n=1):
    
    if (Ny%2 == 0):
        y = np.arange(0,Ny) - Ny/2 + 0.5
    else:
        y = np.arange(0,Ny) - (Ny-1)/2
    
    if (Nx%2 == 0):
        x = np.arange(0,Nx) - Nx/2 + 0.5
    else:
        x = np.arange(0,Nx) - (Nx-1)/2

    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # D = np.sqrt(np.square(X) + np.square(Y))
    D = np.hypot(X, Y)
    
    if filt_type == 'gaussian':
        filter_mask = np.exp(-np.square(D)/(2*np.square(sigma)))

    elif filt_type == 'btw':
        filter_mask = 1/(1+(D/sigma)**(2*n))

    elif filt_type == 'ideal':
        filter_mask = np.ones([Nx, Ny])
        filter_mask[D>sigma] = 0

    else:
        print('Greška! Nije podržan tip filtra: ', filt_type)
        return
    
    # filter_mask = filter_mask / np.max(filter_mask)
    return filter_mask