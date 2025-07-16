import numpy as np

# --- Parameters---
params = {
    "T_start": 0.1,
    "cooling_rate": 0.999,
    "max_iter": 50000,
    "area_threshold": 0.5,
    "epsilon": 0.005,
    "verbose": True,
    "lambda_tv": 0.0003
}

class SimulatedAnnealing:
    def __init__(
            self, objective_function,
            initial_state, neighbour_function, params,
            objective_function_args=None
    ):
        self.objective_function = objective_function
        self.state = initial_state.copy()
        self.neighbour_function = neighbour_function
        self.T_start = params["T_start"]
        self.cooling_rate = params["cooling_rate"]
        self.max_iter = params["max_iter"]
        self.epsilon = params["epsilon"]
        self.verbose = params["verbose"]
        if "lambda_tv" in params:
            self.lambda_tv = params["lambda_tv"]
        self.objective_function_args = objective_function_args if objective_function_args is not None else {}
        self.boundary_recalc_freq = params["boundary_recalc_freq"]
        
    
    def run(self):
        for iteration in range(self.max_iter):
            # Periodično ažuriranje piksela na granici za efikasnije uzorkovanje
            if iteration % self.boundary_recalc_freq == 0:
                X_2d = X.reshape(N, N)

                selem = np.ones((3, 3), dtype=bool) # 3x3 strukturni element za 8-povezanost

                eroded_X = binary_erosion(X_2d, structure=selem)
                dilated_X = binary_dilation(X_2d, structure=selem)

                boundary_mask = (X_2d != eroded_X) | (X_2d != dilated_X)
                temp_boundary_indices = np.where(boundary_mask.ravel())[0].tolist()

                if len(temp_boundary_indices) > 0:
                    current_boundary_indices = temp_boundary_indices
                else:
                    # Rezervni mehanizam: ako nema jasnih granica, vratite se na FBP-ponderisano uzorkovanje
                    # ili nasumično uzorkovanje svih piksela. Uzimamo 10% svih piksela nasumično ponderisano FBP-om.
                    current_boundary_indices = np.random.choice(
                        len(X), p=prob_weights, size=int(len(X)*0.1)
                    ).tolist()
                    if verbose:
                        print(f"[{iteration}] Nema jasne granice, vraćam se na nasumično FBP-ponderisano uzorkovanje.")

            # Odabir piksela za flipovanje: preferiramo granice
            if len(current_boundary_indices) > 0:
                idx = np.random.choice(current_boundary_indices)
            else:
                # Sigurnosna mera: trebalo bi biti pokriveno gornjim fallbackom, ali za svaki slučaj
                idx = np.random.choice(len(X), p=prob_weights)


            # Privremeno flipovanje piksela
            original_val = X[idx]
            X[idx] = 1 - original_val # Flipovanje piksela (0 na 1, ili 1 na 0)

            # Izračunavanje greške za flikovanu sliku koristeći funkciju sa TV penalom
            current_proj_error = proj_error_with_tv(X)

            # --- Simulated Annealing logika za prihvatanje/odbijanje ---
            if current_proj_error < best_err:
                # Uvek prihvati bolji potez
                best_err = current_proj_error
                best_X = X.copy() # Sačuvaj sliku koja je dala bolju grešku
                loss_history.append(best_err) # Sačuvaj poboljšani gubitak
                if verbose:
                    print(f"[{iteration}] T: {current_T:.6f}, Poboljšana greška: {best_err:.6f}")
                if best_err < epsilon:
                    if verbose and iteration % 500 == 0:
                        print(f"Dostignuta željena greška < {epsilon} u iteraciji {iteration}")
                    break # Zaustavi se ako je greška ispod epsilona
            else:
                # Izračunaj verovatnoću prihvatanja lošijeg poteza
                delta_E = current_proj_error - best_err # Razlika u grešci (pozitivna za lošiji potez)
                # Koristimo temperaturu iz trenutne iteracije za SA prihvatanje
                acceptance_probability = np.exp(-delta_E / (current_T + 1e-8)) # Dodato 1e-8 za stabilnost

                if np.random.rand() < acceptance_probability:
                    # Prihvati lošiji potez sa određenom verovatnoćom
                    # NEMA REVERTOVANJA: X ostaje flikovan
                    # ALI: best_X i best_err OSTAJU NEPROMENJENI (drže globalno najbolji)
                    if verbose and iteration % 1000 == 0: # Štampa ređe za lošije poteze
                        print(f"[{iteration}] T: {current_T:.6f}, Prihvaćen lošiji potez. Trenutna greška: {current_proj_error:.6f}")
                else:
                    # Odbaci lošiji potez, vrati piksel na originalnu vrednost
                    X[idx] = original_val
                    
            # Smanji temperaturu (funkcija hlađenja)
            current_T *= cooling_rate

            # Opciono: Postavite minimalnu temperaturu da ne bi pala previše nisko
            # npr. current_T = max(current_T, T_min)
        




# --- Funkcija greške sa Totalnom Varijacijom (TV) regularizacijom ---
def proj_error_with_tv(x_bin_flat, W, sino_target, lambda_tv=0.0003):
    """
    Calculates the total error as the sum of the L2 norm between the actual and synthetic sinogram,
    plus a Total Variation (TV) penalty.
    Parameters
    ----------
    x_bin_flat : np.ndarray
        Flattened binary image array (containing 0s and 1s).
    W : np.ndarray
        Projection matrix used to generate the synthetic sinogram.
    sino_target : np.ndarray
        Target (measured) sinogram.
    lambda_tv : float, optional
        Weight for the Total Variation (TV) penalty term (default is 0.0003).
    Returns
    -------
    float
        The total error: L2 norm of the sinogram difference plus lambda_tv times the TV penalty.
    Notes
    -----
    The TV penalty encourages piecewise constant solutions by penalizing the number of boundaries
    between different pixel values in the reconstructed image.
    """

    # 1. Greška sinograma (L2 norma)
    sino_error = np.linalg.norm(W @ x_bin_flat - sino_target)

    # 2. Penal za Totalnu Varijaciju (TV)
    # Preoblikovanje u 2D za lakše izračunavanje suseda
    N = int(np.sqrt(len(x_bin_flat)))  # Assuming x_bin_flat is a flattened square image
    x_2d = x_bin_flat.reshape(N, N)

    # Horizontalne razlike (broj horizontalnih granica)
    # np.abs(x_2d[:, 1:] - x_2d[:, :-1]) će biti 1 ako su susedi različiti (0-1 ili 1-0), 0 inače.
    # Sumiranjem se broje sve te granice.
    tv_h = np.sum(np.abs(x_2d[:, 1:] - x_2d[:, :-1]))

    # Vertikalne razlike (broj vertikalnih granica)
    tv_v = np.sum(np.abs(x_2d[1:, :] - x_2d[:-1, :]))

    tv_penalty = tv_h + tv_v

    # Ukupna funkcija greške
    return sino_error + lambda_tv * tv_penalty