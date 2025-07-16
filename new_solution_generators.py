import numpy as np
import scipy.ndimage as ndi

class FlipOnEdge:
    def __init__(self, fbp_flat, boundary_recalc_freq, fix_high_distance_pixels=False, fixed_pixels_per_iter=None, max_fix_iter=10):

        prob_weights = fbp_flat - fbp_flat.min()
        prob_weights += 1e-8  # Prevent zeros
        prob_weights /= prob_weights.sum()
        self.prob_weights = prob_weights

        self.boundary_recalc_freq = boundary_recalc_freq

        self.call_count = 0

        self.N = int(np.sqrt(len(fbp_flat)))
        if self.N * self.N != len(fbp_flat):
            raise ValueError("fbp_flat must be a flattened square image (N*N pixels).")

        self.fixed_mask = np.zeros((self.N, self.N), dtype=bool)
        self.fixed_pixels_per_iter = fixed_pixels_per_iter
        self.max_fix_iter = max_fix_iter  
        self.do_fix = fix_high_distance_pixels
        if self.do_fix and self.fixed_pixels_per_iter is None:
            self.fixed_pixels_per_iter = int(0.1 * len(fbp_flat))  # Default to 10% of pixels


    def __call__(self, X):
        # periodically update boundaries
        if self.call_count % self.boundary_recalc_freq == 0:
            self.update_boundaries(X)
            if self.do_fix:
                self.fix_high_distance_pixels()

        # flip one pixel from a boundary
        if len(self.current_boundary_indices) > 0:
            idx = np.random.choice(self.current_boundary_indices)
        else:
            # Sigurnosna mera: trebalo bi biti pokriveno gornjim fallbackom, ali za svaki slučaj
            idx = np.random.choice(len(self.X), p=self.prob_weights)


        neighbour = self.X.copy()
        # Privremeno flipovanje piksela
        neighbour[idx] = 1 - neighbour[idx] # Flipovanje piksela (0 na 1, ili 1 na 0)

        self.call_count += 1
        return neighbour

    def fix_high_distance_pixels(self):
        for _ in range(self.max_fix_iter):
            # --- Fix high-distance pixels cumulatively ---
            fixed_mask = np.zeros_like(self.X.shape, dtype=bool)
            X_2d = self.X.reshape(self.N, self.N)
            dt = ndi.distance_transform_edt(1 - X_2d)
            dt_flat = dt.ravel()
            candidate_idxs = np.argsort(-dt_flat)
            added = 0
            for i in candidate_idxs:
                if not fixed_mask[i] and self.X[i] == 1:
                    fixed_mask[i] = True
                    added += 1
                if added >= self.fixed_pixels_per_iter:
                    break

            self.current_boundary_indices = [i for i in self.current_boundary_indices if not fixed_mask[i]]
            if len(self.current_boundary_indices) > 0:
                break
            else:
                self.current_boundary_indices = self.update_boundaries(self.X)
            


    def update_boundaries(self, X):
        N = int(np.sqrt(X.shape[0]))
        if N * N != X.shape[0]:
            raise ValueError("X must be a flattened square image (N*N pixels).")

        X_2d = X.reshape(N, N)

        selem = np.ones((3, 3), dtype=bool) # 3x3 strukturni element za 8-povezanost

        eroded_X = ndi.binary_erosion(X_2d, structure=selem)
        dilated_X = ndi.binary_dilation(X_2d, structure=selem)

        boundary_mask = (X_2d != eroded_X) | (X_2d != dilated_X)
        temp_boundary_indices = np.where(boundary_mask.ravel())[0].tolist()

        if len(temp_boundary_indices) > 0:
            updated_boundary_indices = temp_boundary_indices
        else:
            # Rezervni mehanizam: ako nema jasnih granica, vratite se na FBP-ponderisano uzorkovanje
            # ili nasumično uzorkovanje svih piksela. Uzimamo 10% svih piksela nasumično ponderisano FBP-om.
            updated_boundary_indices = np.random.choice(
                len(X), p=self.prob_weights, size=int(len(X)*0.1)
            ).tolist()
        

        return updated_boundary_indices
    
class DeterioratingHammingDistance:
    def __init__(
            self,
            max_iter,
            h_min=1,
            h_max=None,
    ):
        self.max_iter = max_iter
        self.h_min = h_min
        self.h_max = h_max
        self.iteration = 0

        

    def __call__(self, X):
        n_bits = len(X)
        if self.h_max is None:
            self.h_max = max(1, int(np.ceil(0.10 * n_bits)))  # sensible default

        # ----- linear schedule ----
        h = ((self.h_min - self.h_max) / (self.max_iter - 1)) * (self.iteration - 1) + self.h_max
        h = int(round(h))
        h = max(self.h_min, h)

        flip_idx = np.random.choice(n_bits, size=h, replace=False)
        x_new = X.copy()
        x_new[flip_idx] = 1 - x_new[flip_idx]           # bit-flip

        self.iteration += 1

        return x_new

