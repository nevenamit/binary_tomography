import argparse
import tempfile
import numpy as np
import skimage as ski
from scipy.sparse.linalg import lsqr
import astra
import astra_wrappers
import utility as util
import os

def main():
    parser = argparse.ArgumentParser(description="ASTRA sinogram + reconstruction demo")
    parser.add_argument('image_path', help='Path to a square grayscale image')
    parser.add_argument('--gpu', action='store_true', help='Use GPU projector if available')
    parser.add_argument('--outdir', default='.', help='Directory to save outputs (default: current directory)')
    args = parser.parse_args()

    # Make sure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    # Load and normalise the input
    img = ski.img_as_float32(ski.io.imread(args.image_path, as_gray=True))
    # binarize image
    img = (img > 0.5).astype(np.float32)

    result = astra_wrappers.preprocess_image(img, show_results=True, angles=[0.0, np.pi/2], M=img.shape[0])
    print(f"Input image shape: {img.shape}")
    print(f"Sinogram shape: {result['sinogram'].shape}")

    # Save outputs
    ski.io.imsave(os.path.join(args.outdir, 'reconstruction.tif'), result["rec_fbp"].astype(np.float32))
    # Binarize the reconstructed image
    rec_binarized = (result["rec_fbp"] > 0.5).astype(np.float32)
    np.savetxt(os.path.join(args.outdir, 'reconstruction_binarized.csv'), rec_binarized, delimiter=',')
    np.savetxt(os.path.join(args.outdir, 'reconstruction.csv'), result["rec_fbp"], delimiter=',')
    np.savetxt(os.path.join(args.outdir, 'sinogram.csv'), result["sinogram"], delimiter=',')
    print(f"Done ✓  Results saved to: {args.outdir}")

if __name__ == '__main__':
    main()