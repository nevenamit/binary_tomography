import argparse
import matplotlib.pyplot as plt

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

    angles = np.linspace(0, np.pi/2, 2)  # 5 angles from 0 to π

    M = 384
    result = astra_wrappers.preprocess_image(img, show_results=True, angles=angles, M=M)
    X0 = astra_wrappers.calculate_X0(result['sinogram'].ravel(), angles, result['rec_art'])

    N = int(np.sqrt(X0.shape[0]))
    X0 = np.reshape(X0, (N, N))
    util.plot_images([X0], ['a'])
    # plt.show()
    
    print(f"Input image shape: {img.shape}")
    print(f"Sinogram shape: {result['sinogram'].shape}")

    # Save outputs
    # xx = ski.color.gray2rgb(X0.astype(np.float32))
    # xx = ski.img_as_ubyte(xx*255)
    ski.io.imsave(os.path.join(args.outdir, 'reconstruction_X0.png'), ski.img_as_ubyte(X0.astype(np.float32)))
    # Binarize the reconstructed image
    np.savetxt(os.path.join(args.outdir, 'reconstruction_binarized.csv'), X0, delimiter=',')
    np.savetxt(os.path.join(args.outdir, 'sinogram.csv'), result["sinogram"], delimiter=',')
    np.savetxt(os.path.join(args.outdir, 'angles.csv'), result["angles"], delimiter=',')
    print(f"Done ✓  Results saved to: {args.outdir}")

if __name__ == '__main__':
    main()