#!/usr/bin/env python3
"""
Compute a 5-angle parallel-beam sinogram and two reconstructions (SART and LSQR-FBP-like)
with ASTRA-toolbox.  Usage:

    python compute_sinogram.py path/to/image.[png|jpg|tif|…]
"""
import argparse
import tempfile
import numpy as np
from skimage import io, img_as_float32
from scipy.sparse.linalg import lsqr
import astra

def make_geometries(N: int):
    """Return (vol_geom, proj_geom, M) for an N×N volume."""
    M = int(2 ** np.ceil(np.log2(N * np.sqrt(2))))  # padding width for projections
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
    return x.reshape(N, N)

def main():
    parser = argparse.ArgumentParser(description="ASTRA sinogram + reconstruction demo")
    parser.add_argument('image_path', help='path to a square grayscale image')
    parser.add_argument('--gpu', action='store_true', help='use GPU projector if available')
    args = parser.parse_args()

    # Load and normalise the input
    img = img_as_float32(io.imread(args.image_path, as_gray=True))
    if img.shape[0] != img.shape[1]:
        raise ValueError("Input image must be square (got shape %s)" % (img.shape,))
    N = img.shape[0]

    # Geometry & projector
    vol_geom, proj_geom, M, angles = make_geometries(N)
    proj_id = astra.create_projector(
        'strip' if not args.gpu else 'strip_fanflat_cuda',
        proj_geom, vol_geom
    )

    # --- Forward projection --------------------------------------------------
    sinogram_id, sinogram = forward_project(img, vol_geom, proj_id)
    np.savetxt('sinogram.csv', sinogram, delimiter=',')

    # --- Reconstructions ------------------------------------------------------
    rec_art  = sart_reconstruction(sinogram_id, vol_geom, proj_id, n_iter=20)
    rec_fbp  = lsqr_fbp_like(sinogram, proj_id, N)

    # Save only the LSQR/FBP-like version as requested
    io.imsave('reconstruction.tif', rec_fbp.astype(np.float32))

    np.savetxt('reconstruction.csv', rec_fbp, delimiter=',')
    import matplotlib.pyplot as plt

    plt.imshow(rec_fbp, cmap='gray')
    plt.title('LSQR/FBP-like Reconstruction')
    plt.axis('off')
    plt.show()

    # Optional: also save the SART volume (uncomment if you want it)
    # io.imsave('reconstruction_sart.tif', rec_art.astype(np.float32))

    # House-keeping
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    print("Done ✓  • sinogram.csv  • reconstruction.png")

if __name__ == '__main__':
    main()