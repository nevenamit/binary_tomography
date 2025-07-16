import numpy as np
import matplotlib.pyplot as plt
import skimage as ski


def plot_images(images  : list,
                titles  : list,
                rows    : int = None,
                cols    : int = None,
                width   : int = 6
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

    width : int, optional
        width of one subplot in inches
    
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
    
    plt.show()

