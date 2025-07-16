import astra_wrappers
import utility as util
import reconstruction_alogrithms
import skimage as ski

import numpy as np

def main():
    img_path = 'data/two_circles.png'
    img = ski.io.imread(img_path, as_gray=True)
    # calculate projections, fbp and sart reconstruction and w matrix
    result = astra_wrappers.preprocess_image(img, show_results=True)

    # --- Parameters---
    params = {
        "T_start": 0.1,
        "cooling_rate": 0.999,
        "max_iter": 50000,
        "area_threshold": 0.5,
        "epsilon": 0.005,
        "verbose": True,
        "lambda_tv": 0.0003,
        "boundary_recalc_freq" : 100
    }

    # Initialize Simulated Annealing
    sino_target = result["sinogram"].ravel()
    X_init = result["rec_fbp"].ravel()
    angles = result["angles"]

    # Estimate area from sinograms
    A = int(np.round(sum(sino_target)/len(angles)))

    fbp_flat = result["rec_fbp"].ravel().astype(np.float64)
    prob_weights = fbp_flat - fbp_flat.min()
    prob_weights += 1e-8  # Prevent zeros
    prob_weights /= prob_weights.sum()

    # take A most shiny pixels
    X = np.zeros_like(X_init, dtype=np.uint8)
    X[np.argsort(X_init)[-A:]] = 1


    SA = reconstruction_alogrithms.SimulatedAnnealing(
        objective_function=reconstruction_alogrithms.proj_error_with_tv,
        initial_state=X,
        neighbour_function=reconstruction_alogrithms.neighbour_function_edges,
        params=params,
        objective_function_args=(
            result["system_matrix"], 
            result["sinogram"].ravel(),
            params["lambda_tv"]
        )
    )





if __name__ == "__main__":
    main()