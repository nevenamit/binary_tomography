import numpy as np
import scipy.ndimage as ndi
import utility as util

def morph_open_close(image, structure_size=3, plot=False):

    morph_structure = np.ones((structure_size, structure_size), dtype=bool) # 3x3 square
    closed = ndi.binary_closing(image, structure=morph_structure, iterations=1)
    opened = ndi.binary_opening(closed, structure=morph_structure, iterations=1)

    if plot:

        util.plot_images(
            [image, closed, opened],
            titles=['Original Image', 'closed Image', 'opened Image']
        )

    return opened


def proj_by_proj_bin_refinement(X, W, sino_target, epsilon_rel=0.01, max_passes=3):
    num_projs = W.shape[0]
    all_projection_indices = np.arange(num_projs)

    for pass_idx in range(max_passes):
        changes = 0

        np.random.shuffle(all_projection_indices)

        for i in all_projection_indices:
            row = W.getrow(i)
            p_real = sino_target[i]
            p_sim = row @ X
            diff = p_sim.item() - p_real

            if abs(diff) <= epsilon_rel * (abs(p_real) + 1e-8):
                continue  # already within tolerance

            # Pixels involved in this projection
            involved_pixels = row.indices
            if len(involved_pixels) == 0:
                continue  # skip empty rows

            if diff > 0:
                candidates = [j for j in involved_pixels if X[j] == 1]
            else:
                candidates = [j for j in involved_pixels if X[j] == 0]

            if len(candidates) == 0:
                continue

            # Greedy pick: one that reduces error the most
            best_j = None
            best_err = abs(diff)
            original_proj = p_sim.item()

            for j in candidates:
                X[j] = 1 - X[j]  # flip
                new_proj = row @ X
                new_err = abs(new_proj.item() - p_real)
                if new_err < best_err:
                    best_j = j
                    best_err = new_err
                X[j] = 1 - X[j]  # revert

            if best_j is not None and best_err < abs(diff):
                X[best_j] = 1 - X[best_j]
                changes += 1

        print(f"Pass {pass_idx+1}: {changes} pixels updated")
        if changes == 0:
            break  # converged

    # Final image
    refined = X.reshape(N, N)
    return refined







