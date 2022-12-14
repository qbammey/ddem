from pathlib import Path

import numpy as np
from skimage.util import view_as_blocks, view_as_windows
import imageio
import scipy as sp
import scipy.stats

from demosaic import demosaic


def get_maps(img_path, algs):
    img_path = Path(img_path)
    img = imageio.imread(img_path)[:, :, :3] / 255
    Y, X, _ = img.shape
    residuals = np.zeros((len(algs), 4, Y, X, 3))
    for grid in range(4):
        for i_alg, alg in enumerate(algs):
            residuals[i_alg, grid] = demosaic(img_path, alg, grid) / 255 - img
    Y -= Y % 2
    X -= X % 2
    residuals = np.mean(np.square(residuals[:, :, :Y, :X]), axis=-1)
    residuals = view_as_blocks(residuals, (len(algs), 4, 2, 2))[0, 0]
    residuals = residuals.mean(axis=(-1, -2))
    grid_estimates = np.argmin(np.min(residuals, axis=2), axis=2)
    alg_estimates = np.argmin(np.min(residuals, axis=3), axis=2)
    return grid_estimates, alg_estimates


def find_inconsistencies_grid(grid_estimates, W):
    grid_estimates_windowed = view_as_windows(grid_estimates, (W, W))
    diag_estimates = grid_estimates // 2
    diag_estimates_windowed = grid_estimates_windowed // 2
    diag_global = sp.stats.mode(diag_estimates.ravel()).mode[0]
    diag_global_pvalue = sp.stats.binom.sf(np.count_nonzero(diag_estimates == diag_global), diag_estimates.size, 0.5)
    diag_n_incorrect_per_window = np.count_nonzero(diag_estimates_windowed != diag_global, axis=(-1, -2))
    diag_n_correct_per_window = np.count_nonzero(diag_estimates_windowed == diag_global, axis=(-1, -2))
    inconsistent_diag_nfa = diag_estimates.size / (W ** 2) * sp.stats.binom.sf(diag_n_incorrect_per_window, W ** 2, 0.5)
    consistent_diag_nfa = diag_estimates.size / (W ** 2) * sp.stats.binom.sf(diag_n_correct_per_window, W ** 2, 0.5)
    #
    grid_global = sp.stats.mode(grid_estimates.ravel()[diag_estimates.ravel() == diag_global]).mode[0]
    grid_global_opposite = [1, 0, 3, 2][grid_global]
    grid_global_pvalue = sp.stats.binom.sf(np.count_nonzero(grid_estimates == grid_global),
                                           np.count_nonzero(diag_estimates == diag_global), 0.5)
    count_is_diag_global = np.count_nonzero(diag_estimates_windowed == diag_global, axis=(-1, -2))
    n_tests = grid_estimates.size / (W ** 2)
    diag_is_correct_windowed = diag_estimates_windowed == diag_global
    grid_is_global_windowed = grid_estimates_windowed == grid_global
    grid_is_opposite_windowed = grid_estimates_windowed == grid_global_opposite
    inconsistent_grid_nfa = n_tests * sp.stats.binom.sf(
        np.count_nonzero(diag_is_correct_windowed*grid_is_opposite_windowed, axis=(-1, -2))-1,
        count_is_diag_global, 0.5)
    consistent_grid_nfa = n_tests * sp.stats.binom.sf(
        np.count_nonzero(diag_is_correct_windowed*grid_is_global_windowed, axis=(-1, -2))-1,
        count_is_diag_global, 0.5)
    return inconsistent_diag_nfa, inconsistent_grid_nfa, diag_global_pvalue, grid_global_pvalue, diag_global, grid_global, consistent_diag_nfa, consistent_grid_nfa


def find_inconsistencies_alg(alg_estimates, n_algs, W):
    alg_global = sp.stats.mode(alg_estimates.ravel()).mode[0]
    alg_global_pvalue = sp.stats.binom.sf(np.count_nonzero(alg_estimates == alg_global)-1, alg_estimates.size, 1 / n_algs)
    alg_windowed = view_as_windows(alg_estimates, (W, W))
    alg_estimates_mode = sp.stats.mode(alg_windowed.reshape(alg_windowed.shape[0], alg_windowed.shape[1], W * W),
                                       axis=2)
    alg_is_incorrect = alg_estimates_mode.mode[:, :,
                       0] != alg_global  # incorrect as in different from the global estimate
    alg_is_correct = alg_estimates_mode.mode[:, :, 0] == alg_global
    alg_local_estimate_counts = alg_estimates_mode.count[:, :, 0]
    consistent_alg_nfa = alg_estimates.size / (W ** 2) * sp.stats.binom.sf(alg_local_estimate_counts * alg_is_correct,
                                                                           W * W, 1 / n_algs)
    inconsistent_pvalue_1 = sp.stats.binom.sf(alg_local_estimate_counts * alg_is_incorrect - 1, W * W, 1 / n_algs)
    alg_is_correct_count = np.sum(alg_windowed == alg_global, axis=(-1, -2))
    inconsistent_pvalue_2 = sp.stats.binom.sf(alg_local_estimate_counts * alg_is_incorrect - 1, alg_local_estimate_counts + alg_is_correct_count, 0.5)
    # 1: the incorrect alg is significant compared to the pool of all possible algorithms, 2: it is significant compared to the global alg
    inconsistent_alg_nfa = alg_estimates.size / (W ** 2) * np.maximum(inconsistent_pvalue_1, inconsistent_pvalue_2)
    return inconsistent_alg_nfa, alg_global_pvalue, alg_global, consistent_alg_nfa


def register_results(original_image, inconsistency_map, W):
    Y, X = original_image.shape[:2]
    scaled_map = inconsistency_map.repeat(2, axis=0).repeat(2, axis=1)
    padded = np.pad(scaled_map, [(W, W), (W, W)])
    cropped = padded[:Y, :X]
    return cropped


def detect_forgeries(img_path, algs, W, threshold):
    grid_estimates, alg_estimates = get_maps(img_path, algs)
    inconsistent_diag_nfa, inconsistent_grid_nfa, diag_global_pvalue, grid_global_pvalue, diag_global, grid_global, consistent_diag_nfa, consistent_grid_nfa = find_inconsistencies_grid(
        grid_estimates, W)
    inconsistent_alg_nfa, alg_global_pvalue, alg_global, consistent_alg_nfa = find_inconsistencies_alg(alg_estimates,
                                                                                                       len(algs), W)
    n_algs = len(algs)
    multiplier = 3 if n_algs > 1 else 2
    diag_global_pvalue = multiplier * diag_global_pvalue  # as we are doing 3 tests
    grid_global_pvalue = max(diag_global_pvalue,
                             multiplier * grid_global_pvalue)  # if the diagonal is not detected, the complete grid is irrelevant
    alg_global_pvalue = multiplier * alg_global_pvalue
    inconsistent_diag_nfa = np.maximum(multiplier * inconsistent_diag_nfa, diag_global_pvalue)
    inconsistent_grid_nfa = np.maximum(multiplier * inconsistent_grid_nfa, grid_global_pvalue)
    inconsistent_alg_nfa = np.maximum(multiplier * inconsistent_alg_nfa, alg_global_pvalue)
    consistent_diag_nfa = np.maximum(multiplier * consistent_diag_nfa, diag_global_pvalue)
    consistent_grid_nfa = np.maximum(multiplier * consistent_grid_nfa, grid_global_pvalue)
    consistent_alg_nfa = np.maximum(multiplier * consistent_alg_nfa, alg_global_pvalue)


    detected_inconsistencies_diag = inconsistent_diag_nfa < threshold
    detected_inconsistencies_grid = inconsistent_grid_nfa < threshold
    detected_inconsistencies_pattern = detected_inconsistencies_diag + detected_inconsistencies_grid
    detected_inconsistencies_alg = inconsistent_alg_nfa < threshold
    detected_inconsistencies_merged = detected_inconsistencies_pattern + detected_inconsistencies_alg

    detected_is_consistent_diag = consistent_diag_nfa < threshold
    detected_is_consistent_grid = consistent_grid_nfa < threshold
    detected_is_consistent_alg = consistent_alg_nfa < threshold

    img = imageio.imread(img_path)[:, :, :3]
    Y, X = img.shape[:2]

    # out_diag, out_grid, out_pattern, out_alg, out_merged = [register_results(img, detection, W) for detection in (
    # detected_inconsistencies_diag, detected_inconsistencies_grid, detected_inconsistencies_pattern,
    # detected_inconsistencies_alg, detected_inconsistencies_merged)]

    print_main = []
    print_warning = []
    print_supplementary = []
    local_fail = False
    PRINT_RATIO_THRESHOLD = 0.8
    if alg_global_pvalue > threshold and diag_global_pvalue > threshold:
        consistent = False
        print_main.append(
            f"No information about the demosaicing could be confidently retrieved at a global scale. As such, no detection could be made (Pattern NFA score {diag_global_pvalue:.2e} and algorithm NFA score {alg_global_pvalue:.2e} are both above the tolerated false alarms rate threshold {threshold:.2e}.")
    elif np.any(detected_inconsistencies_merged):
        consistent = False
        print_main.append("Inconsistencies were detected. The image is considered forged by this method:")
        if np.any(detected_inconsistencies_pattern):
            ratio_inconsistent = np.count_nonzero(
                detected_inconsistencies_pattern) / detected_inconsistencies_pattern.size
            most_significant_nfa = min(inconsistent_diag_nfa.min(), inconsistent_grid_nfa.min())
            print_supplementary.append(
                f"Inconsistencies on the demosaicing pattern were detected in {ratio_inconsistent:.0%} of the windows. The most significant detection had a NFA score of {most_significant_nfa:.2e}.")
        if np.any(detected_inconsistencies_alg):
            ratio_inconsistent = np.count_nonzero(detected_inconsistencies_alg) / detected_inconsistencies_alg.size
            most_significant_nfa = inconsistent_alg_nfa.min()
            print_supplementary.append(
                f"Inconsistencies on the estimated demosaicing algorithm were detected in {ratio_inconsistent:.0%} of the windows. The most significant detection had a NFA score of {most_significant_nfa:.2e}. Note that algorithm estimation detections are less reliable than pattern shift detections.")
        print_supplementary.append("See the image output for localization of the detected forgeries.")
    else:
        consistent = True
        print_main.append("No traces of forgery were detected.")
        # No forgery detection
        if diag_global_pvalue > threshold:
            print_warning.append(
                f"No information about the demosaicing pattern could be confidently retrieved at a global scale. As such, no detection could be made based on the demosaicing pattern estimation (NFA score {diag_global_pvalue:.2e} is above the tolerated false alarms rate threshold {threshold:.2e}.")
        else:
            print_supplementary.append(
                f"The diagonal of the demosaicing pattern could be confidently estimated globally as {diag_global} (NFA score {diag_global_pvalue:.2e} is below the tolerated false alarms rate threshold {threshold:.2e}.")
            if not np.any(detected_is_consistent_diag):
                local_fail = True
                print_warning.append(
                    "Although the diagonal of the demosaicing pattern could be estimated globally, it was not retrieved locally on any of the windows")
                local_diagonals_detected = False
            else:
                local_diagonals_detected = True
                if np.all(detected_is_consistent_diag):
                    if grid_global_pvalue >= threshold or not np.all(detected_is_consistent_grid):
                        # otherwise, it means the full pattern is detected everywhere, and the following line would be redundant
                        print_supplementary.append(
                            "The diagonal of the demosaicing pattern was confidently retrieved in all tested windows.")
                else:
                    ratio = np.mean(detected_is_consistent_diag)
                    if ratio > PRINT_RATIO_THRESHOLD:
                        print_supplementary.append(
                            f"The diagonal of the demosaicing pattern was confidently retrieved in {ratio:.0%} of the tested windows.")
                    else:
                        local_fail = True
                        print_warning.append(
                            f"The diagonal of the demosaicing pattern was only confidently retrieved in {ratio:.0%} of the tested windows.")
            # now test for full grid
            if grid_global_pvalue > threshold:
                print_warning.append(
                    f"Although the diagonal pattern was found globally, the complete pattern could not be retrieved (NFA score {grid_global_pvalue:.2e} is above the tolerated false alarms rate threshold {threshold:.2e}.")
            else:
                print_supplementary.append(
                    f"The complete demosaicing pattern could be estimated globally as {grid_global} (The diagonal was estimated with a NFA score {diag_global_pvalue:.2e} and the full pattern with a NFA score {grid_global_pvalue:.2e} are both below the tolerated false alarms rate threshold {threshold:.2e}.")
                if local_diagonals_detected:
                    if not np.any(detected_is_consistent_grid):
                        local_fail = True
                        print_warning.append(
                            "Although the full demosaicing pattern was estimated globally, it could not be confidently retrieved locally in any of the windows.")
                    elif np.all(detected_is_consistent_grid):
                        print_supplementary.append(
                            "The full demosaicing pattern was confidently retrieved in all tested windows.")
                    else:
                        ratio = np.mean(detected_is_consistent_grid)
                        if ratio > PRINT_RATIO_THRESHOLD:
                            print_supplementary.append(
                                f"The full demosaicing pattern was confidently retrieved in {ratio:.0%} of the tested windows.")
                        else:
                            local_fail = True
                            print_warning.append(
                                f"The full demosaicing pattern was only confidently retrieved in {ratio:.0%} of the tested windows.")
            # finally, test for algorithms
            if alg_global_pvalue > threshold and n_algs > 1:
                print_warning.append(
                    f"No information about the demosaicing algorithm could be confidently retrieved at a global scale. As such, no detection could be made based on the algorithm estimation (NFA score {alg_global_pvalue:.2e} is above the tolerated false alarms rate threshold {threshold:.2e}. Better results might be obtained by including more demosaicing algorithms.")
            else:
                if (not np.any(detected_is_consistent_alg)) and n_algs > 1:
                    local_fail = True
                    print_warning.append(
                        "Although the closest demosaicing algorithm was estimated globally, it could not be confidently retrieved locally in any of the windows")
                elif np.all(detected_is_consistent_alg) and n_algs > 1:
                    print_supplementary.append(
                        "The closest demosaicing algorithm was confidently retrieved in all tested windows")
                elif n_algs > 1:
                    ratio = np.mean(detected_is_consistent_alg)
                    if ratio > PRINT_RATIO_THRESHOLD:
                        print_supplementary.append(
                            f"The closest demosaicing algorithm was confidently retrieved in {ratio:.0%} of the tested windows.")
                    else:
                        local_fail = True
                        print_warning.append(
                            f"The closest demosaicing algorithm was only confidently retrieved in {ratio:.0%} of the tested windows.")
    if n_algs == 1:
        print_warning.append("Only one demosaicing algorithm has been selected for the comparison. The method was thus only run for pattern inconsistencies detection.")

    for line in print_main:
        print(line)
    if consistent:
        if local_fail or len(print_warning) > 0:
            print("However:")
            if local_fail:
                print(
                    "Some information could not be fully retrieved locally. Increasing the window size might help, or it might also be that some regions are too saturated for analysis to be possible.")
            for line in print_warning:
                print(line)
        print(
            "\nNote that even in regions where the local parameters coincide with the global detection, the absence of forgery detection is not an absolute proof that the image is not forged. The demosaicing information might have been preserved by the manipulation, the mosaic might even have been artificially reconstructed after the manipulation, or the forgery may even simply be too small compared to the window size.")
        if not local_fail:
            print(
                "The method has found significant results in most windows. You may want to try decreasing the window size to refine the analysis.")

    for line in print_supplementary:
        print(line)


    detected_inconsistencies_diag, detected_inconsistencies_grid, detected_inconsistencies_pattern, detected_inconsistencies_alg, detected_inconsistencies_merged, detected_is_consistent_diag, detected_is_consistent_grid, detected_is_consistent_alg = [
        register_results(img, detection, W) for detection in (
        detected_inconsistencies_diag, detected_inconsistencies_grid, detected_inconsistencies_pattern, detected_inconsistencies_alg,
        detected_inconsistencies_merged, detected_is_consistent_diag, detected_is_consistent_grid,
        detected_is_consistent_alg)]
    return detected_inconsistencies_diag, detected_inconsistencies_grid, detected_inconsistencies_pattern, detected_inconsistencies_alg, detected_inconsistencies_merged, detected_is_consistent_diag, detected_is_consistent_grid, detected_is_consistent_alg