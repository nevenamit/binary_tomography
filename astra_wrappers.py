import argparse
import tempfile
import numpy as np
import skimage as ski
from scipy.sparse.linalg import lsqr
import astra
import utility as util

def create_complex_binary_phantom(size=256):
  """Creates a complex binary phantom with diverse shapes.

  Args:
    size: The size of the square phantom (e.g., 256 for 256x256).

  Returns:
    A 2D numpy array representing the binary phantom.
  """
  phantom = np.zeros((size, size), dtype=np.uint8)

  # Add some simple shapes
  # Square
  phantom[int(size * 0.2):int(size * 0.4), int(size * 0.2):int(size * 0.4)] = 1
  # Circle
  center_x, center_y = int(size * 0.6), int(size * 0.3)
  radius = int(size * 0.15)
  y, x = np.ogrid[:size, :size]
  dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
  phantom[dist_from_center <= radius] = 1
  # Rectangle
  phantom[int(size * 0.7):int(size * 0.8), int(size * 0.5):int(size * 0.9)] = 1

  '''
  # Add more complex shapes using random walks or cellular automata for irregularity
  # This is a simplified example using random points and smoothing
  num_irregular_shapes = 3
  for _ in range(num_irregular_shapes):
    start_x, start_y = np.random.randint(0, size, 2)
    shape_size = np.random.randint(int(size * 0.05), int(size * 0.2))
    for i in range(shape_size):
        for j in range(shape_size):
            if np.random.rand() > 0.7: # Introduce some randomness
                x_offset, y_offset = np.random.randint(-5, 6, 2)
                nx, ny = start_x + i + x_offset, start_y + j + y_offset
                if 0 <= nx < size and 0 <= ny < size:
                    phantom[nx, ny] = 1
  '''
  # Optional: Apply a small smoothing filter to make the irregular shapes less blocky
  # from scipy.ndimage import gaussian_filter
  # phantom = gaussian_filter(phantom.astype(float), sigma=1.5) > 0.5

  return phantom.astype(np.uint8)

def make_geometries(N: int, angles=None):
    """Return (vol_geom, proj_geom, M) for an N×N volume."""
    M = int(2 ** np.ceil(np.log2(N * np.sqrt(2))))  # padding width for projections
    if angles is None:
        angles = np.random.rand(5) * np.pi             # 5 random angles in [0, π)
    vol_geom  = astra.create_vol_geom(N, N)
    proj_geom = astra.create_proj_geom('parallel', 1.0, M, angles)
    return vol_geom, proj_geom, M, angles

def forward_project(img, vol_geom, proj_id):
    """Return sinogram (numpy array) and its ASTRA id."""
    # Copy image into ASTRA volume
    img_id = astra.data2d.create('-vol', vol_geom, img)
    sinogram_id, sinogram = astra.create_sino(img_id, proj_id)
    astra.data2d.delete(img_id)
    return sinogram_id, sinogram

def sart_reconstruction(sinogram_id, vol_geom, proj_id, n_iter=20):
    """SART (ART) reconstruction; returns numpy array."""
    rec_id = astra.data2d.create('-vol', vol_geom, 0)  # zeros
    cfg = astra.astra_dict('SART')
    cfg.update({
        'ReconstructionDataId': rec_id,
        'ProjectionDataId'    : sinogram_id,
        'ProjectorId'         : proj_id,
    })
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, n_iter)
    rec = astra.data2d.get(rec_id)
    # clean-up
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    return rec

def lsqr_fbp_like(sinogram, proj_id, N):
    """Solve W·x = p with LSQR ⇒ FBP-like recon; returns (N×N) array."""
    matrix_id = astra.projector.matrix(proj_id)
    W = astra.matrix.get(matrix_id)         # sparse CSR matrix
    x = lsqr(W, sinogram.ravel())[0]
    return x.reshape(N, N), W




def preprocess_image(
        image,
        show_results = False,
        angles=None
    ):

    # check that image is valid
    img = ski.img_as_float32(image)
    if img.shape[0] != img.shape[1]:
        raise ValueError("Input image must be square (got shape %s)" % (img.shape,))
    N = img.shape[0]

    # Geometry & projector
    vol_geom, proj_geom, M, angles = make_geometries(N, angles)
    proj_id = astra.create_projector(
        'strip',
        proj_geom, vol_geom
    )

    # --- Forward projection --------------------------------------------------
    sinogram_id, sinogram = forward_project(img, vol_geom, proj_id)

    # --- Reconstructions ------------------------------------------------------
    rec_art  = sart_reconstruction(sinogram_id, vol_geom, proj_id, n_iter=20)
    rec_fbp, W  = lsqr_fbp_like(sinogram, proj_id, N)


    if show_results == True:
        # Plot and show results
        util.plot_images(
            [img, sinogram, rec_art, rec_fbp], 
            ["original image", "sinogram", "sart reconstruction", "least squares reconstruction"], 
            width=3
        )

    # House-keeping
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    # Return relevant data for further use
    result = {
        "phantom": img,
        "sinogram": sinogram,
        "rec_art": rec_art,
        "rec_fbp": rec_fbp,
        "system_matrix": W,
        "angles" : angles
    }

    return result

    