import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation

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
            X0, neighbour_function, params,
            objective_function_args=None,
            neighbour_function_args=None
    ):
        self.objective_function = objective_function
        self.X = X0.copy()
        self.neighbour_function = neighbour_function
        self.T = params["T_start"]
        self.cooling_rate = params["cooling_rate"]
        self.max_iter = params["max_iter"]
        self.epsilon = params["epsilon"]
        self.verbose = params["verbose"]
        if "lambda_tv" in params:
            self.lambda_tv = params["lambda_tv"]
        self.objective_function_args = objective_function_args if objective_function_args is not None else {}
        self.neighbour_function_args = neighbour_function_args if neighbour_function_args is not None else {}

        if "boundary_recalc_freq" in params:
            self.boundary_recalc_freq = params["boundary_recalc_freq"]
        

        self.N = int(np.sqrt(self.X.shape[0])) # Assuming X0 is a flattened square image
        if self.N * self.N != self.X.shape[0]:
            raise ValueError("X0 must be a flattened square image (N*N pixels).")

        self.best_cost = self.objective_function(self.X, *self.objective_function_args)
        self.current_cost = self.best_cost
        self.best_X = self.X.copy()

        self.cost_history = [self.best_cost]
        self.temp_history = [self.T]

        # ovo je malo lose, ali nemam sad vremena da smisljam bolje smiley face
        self.prob_weights = None 

        if self.verbose:
            plt.ion()
            fig, ax = plt.subplots(2, 2, figsize=(5, 5))
            self.fig = fig
            self.ax = ax.ravel()

            self.line_plot, = self.ax[0].plot([], [])
            self.ax[0].set_xlim(0, self.max_iter)
            self.ax[0].set_ylim(0, self.best_cost * 1.2)
            self.ax[0].set_xlabel("Iteration")
            self.ax[0].set_ylabel("Cost")
            self.ax[0].set_title("Simulated Annealing Progress")

            self.img_plot = self.ax[1].imshow(self.X.reshape(self.N, self.N), cmap='gray', vmin=0, vmax=1)
            self.ax[1].set_title("Current Image")
            self.ax[1].axis('off')

            self.ax[2].set_xlim(0, self.max_iter)
            self.ax[2].set_ylim(0, self.T * 1.2)
            self.ax[2].set_xlabel("Iteration")
            self.ax[2].set_ylabel("Temperature")
            self.ax[2].set_title("Temperature Decay")
            self.temp_plot, = self.ax[2].plot([], [])


            plt.tight_layout()
            plt.show()


    def update_boundaries(self):

        X_2d = self.X.reshape(self.N, self.N)

        selem = np.ones((3, 3), dtype=bool) # 3x3 strukturni element za 8-povezanost

        eroded_X = binary_erosion(X_2d, structure=selem)
        dilated_X = binary_dilation(X_2d, structure=selem)

        boundary_mask = (X_2d != eroded_X) | (X_2d != dilated_X)
        temp_boundary_indices = np.where(boundary_mask.ravel())[0].tolist()

        if len(temp_boundary_indices) > 0:
            self.current_boundary_indices = temp_boundary_indices
        else:
            # Rezervni mehanizam: ako nema jasnih granica, vratite se na FBP-ponderisano uzorkovanje
            # ili nasumično uzorkovanje svih piksela. Uzimamo 10% svih piksela nasumično ponderisano FBP-om.
            self.current_boundary_indices = np.random.choice(
                len(self.X), p=self.prob_weights, size=int(len(self.X)*0.1)
            ).tolist()
            if self.verbose:
                print(f"[{self.iteration}] Nema jasne granice, vraćam se na nasumično FBP-ponderisano uzorkovanje.")


    def generate_neighbour_edges(self, fbp_flat):
        if self.prob_weights is None:
            prob_weights = fbp_flat - fbp_flat.min()
            prob_weights += 1e-8  # Prevent zeros
            prob_weights /= prob_weights.sum()
            self.prob_weights = prob_weights

        # Periodično ažuriranje piksela na granici za efikasnije uzorkovanje
        if self.iteration % self.boundary_recalc_freq == 0:
            self.update_boundaries()

        # Odabir piksela za flipovanje: preferiramo granice
        if len(self.current_boundary_indices) > 0:
            idx = np.random.choice(self.current_boundary_indices)
        else:
            # Sigurnosna mera: trebalo bi biti pokriveno gornjim fallbackom, ali za svaki slučaj
            idx = np.random.choice(len(self.X), p=self.prob_weights)


        neighbour = self.X.copy()
        # Privremeno flipovanje piksela
        neighbour[idx] = 1 - neighbour[idx] # Flipovanje piksela (0 na 1, ili 1 na 0)

        return neighbour


    
    def run(self):
        for self.iteration in range(self.max_iter):

            neighbour = self.neighbour_function(self, *self.neighbour_function_args)
            neighbour_cost = self.objective_function(self.X, *self.objective_function_args)

            if neighbour_cost < self.current_cost:
                # always accept better solution
                self.current_cost = neighbour_cost
                self.X = neighbour
                if self.verbose:
                    print(f"[{self.iteration}] T: {self.T:.6f}, Poboljšana greška: {self.best_cost:.6f}")

                # check if end
                if self.best_cost < self.epsilon:
                    if self.verbose and self.iteration % 500 == 0:
                        print(f"Dostignuta željena greška < {self.epsilon} u iteraciji {self.iteration}")
                    break 

            else:
                # accept worse solution with some probability
                delta_E = neighbour_cost - self.current_cost # Razlika u grešci (pozitivna za lošiji potez)
                # Koristimo temperaturu iz trenutne iteracije za SA prihvatanje
                acceptance_probability = np.exp(-delta_E / (self.T + 1e-8)) # Dodato 1e-8 za stabilnost

                if np.random.rand() < acceptance_probability:
                    # Prihvati lošiji potez sa određenom verovatnoćom
                    # NEMA REVERTOVANJA: X ostaje flikovan
                    # ALI: best_X i best_err OSTAJU NEPROMENJENI (drže globalno najbolji)
                    self.X = neighbour
                    self.current_cost = neighbour_cost
                    if self.verbose and self.iteration % 1000 == 0: # Štampa ređe za lošije poteze
                        print(f"[{self.iteration}] T: {self.T:.6f}, Prihvaćen lošiji potez. Trenutna greška: {neighbour_cost:.6f}")
                    

            if self.current_cost < self.best_cost:
                self.best_cost = self.current_cost
                self.best_X = self.X.copy() # Sačuvaj sliku koja je dala bolju grešku

            if self.verbose and self.iteration % 1000 == 0:
                self.line_plot.set_data(range(self.iteration + 1), self.cost_history)
                self.ax[0].set_xlim(0, self.iteration+1)
                self.ax[0].set_ylim(min(self.cost_history)*0.8, max(self.cost_history) * 1.2)

                self.ax[1].imshow(self.X.reshape(self.N, self.N), cmap='gray', vmin=0, vmax=1)
                self.img_plot.set_data(self.X.reshape(self.N, self.N))

                self.temp_plot.set_data(range(self.iteration+1), self.temp_history)
                self.ax[2].set_xlim(0, self.iteration+1)


                plt.pause(0.01)

            # Smanji temperaturu (funkcija hlađenja)
            self.T *= self.cooling_rate
            self.cost_history.append(self.current_cost)
            self.temp_history.append(self.T)

        if self.verbose:
            plt.ioff()
            plt.show()

        return self.best_X.reshape(self.N, self.N), self.best_cost, self.cost_history




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