import astra_wrappers
import utility as util
import reconstruction_alogrithms
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import cost_functions
import new_solution_generators


def main():
    img_path = 'data/phantom_sz/ph1.png'
    img = ski.io.imread(img_path, as_gray=True)
    # calculate projections, fbp and sart reconstruction and w matrix
    result = astra_wrappers.preprocess_image(img, show_results=True)

    # --- Parameters---
    params = {
        "T_start": 0.1,
        "cooling_rate": 0.99,
        "max_iter": 50000,
        "area_threshold": 0.5,
        "epsilon": 0.00001,
        "verbose": True,
        "lambda_tv": 0.000,
        "boundary_recalc_freq" : 100
    }

    # Initialize Simulated Annealing
    sino_target = result["sinogram"].ravel()
    X_init = result["rec_fbp"].ravel()
    angles = result["angles"]

    # Estimate area from sinograms
    A = int(np.round(sum(sino_target)/len(angles)))
    # take A highest value pixels
    X = np.zeros_like(X_init, dtype=np.uint8)
    X[np.argsort(X_init)[-A:]] = 1

    neighbour_function = new_solution_generators.FlipOnEdge(
        fbp_flat=result["rec_fbp"].ravel(),
        boundary_recalc_freq=params["boundary_recalc_freq"]
    )

    SA = reconstruction_alogrithms.SimulatedAnnealing(
        X0=X,
        cost_function=reconstruction_alogrithms.proj_error_with_tv,
        cost_function_args=(
            result["system_matrix"], 
            result["sinogram"].ravel(),
            params["lambda_tv"]
        ),
        neighbour_function=neighbour_function,
        neighbour_function_args=None,
        params=params,
    )

    best_img, best_cost, cost_history = SA.run()


if __name__ == "__main__":
    main()