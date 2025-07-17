import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


class RandomSearch:
    def __init__(
        self,
        X0,
        cost_function, neighbour_function,
        params,
        cost_function_args=None,
        neighbour_function_args=None
    ):
        self.X = X0.copy()
        self.cost_function = cost_function
        self.cost_function_args = cost_function_args if cost_function_args is not None else ()
        self.neighbour_function = neighbour_function
        self.neighbour_function_args = neighbour_function_args if neighbour_function_args is not None else ()

        self.max_iter = params["max_iter"]
        self.verbose = params["verbose"]


        self.current_cost = self.cost_function(self.X, *self.cost_function_args)
        self.cost_history = [self.current_cost]
        self.iteration = 0

    def run(self):
        for self.iteration in range(self.max_iter):
            neighbour = self.neighbour_function(self.X, *self.neighbour_function_args)
            neighbour_cost = self.cost_function(neighbour, *self.cost_function_args)

            if neighbour_cost < self.current_cost:
                # always accept better solution
                self.current_cost = neighbour_cost
                self.X = neighbour
                if self.verbose:
                    print(f"[{self.iteration}] Poboljšana greška: {self.best_cost:.6f}")

                if self.current_cost < self.epsilon:
                    if self.verbose:
                        print(f"Dostignuta željena greška < {self.epsilon} u iteraciji {self.iteration}")
                    break

            self.cost_history.append(self.current_cost)

        return self.X.reshape(self.N, self.N), self.current_cost, self.cost_history



class SimulatedAnnealing:
    def __init__(
            self, 
            X0,
            cost_function, neighbour_function, 
            params,
            cost_function_args=None,
            neighbour_function_args=None
    ):
        self.cost_function = cost_function
        self.X = X0.copy()
        self.neighbour_function = neighbour_function
        if params["T_start"] is not None:
            self.T = params["T_start"]
        self.cooling_rate = params["cooling_rate"]
        self.max_iter = params["max_iter"]
        self.epsilon = params["epsilon"]
        self.verbose = params["verbose"]
        self.cost_function_args = cost_function_args if cost_function_args is not None else ()
        self.neighbour_function_args = neighbour_function_args if neighbour_function_args is not None else ()

        self.N = int(np.sqrt(self.X.shape[0])) # Assuming X0 is a flattened square image
        if self.N * self.N != self.X.shape[0]:
            raise ValueError("X0 must be a flattened square image (N*N pixels).")

        self.best_cost = self.cost_function(self.X, *self.cost_function_args)
        self.current_cost = self.best_cost
        self.best_X = self.X.copy()

        self.cost_history = [self.current_cost]
        self.temp_history = [self.T]

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

    def estimate_starting_temperature(self, iterations=100):
        costs = []
        for _ in range(iterations):
            x_rand = np.random.randint(0, 2, size=self.N**2)
            cost = self.cost_function(x_rand, *self.cost_function_args)
            costs.append(cost)
        self.T = np.mean(costs) 
        if self.verbose:
            print(f"Estimated starting temperature: {self.T:.6f}")

    
    def run(self):
        for self.iteration in range(self.max_iter):

            neighbour = self.neighbour_function(self.X, *self.neighbour_function_args)
            neighbour_cost = self.cost_function(neighbour, *self.cost_function_args)

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
                acceptance_probability = np.exp(-delta_E / (self.T + 1e-14)) # Dodato 1e-8 za stabilnost

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

        return self.best_X.reshape(self.N, self.N), self.best_cost, self.cost_history, self.temp_history
