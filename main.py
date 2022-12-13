#!/usr/bin/env python
import argparse
import imageio
import numpy as np

from analysis import detect_forgeries

ALGS = ['bilinear', 'gunturk', 'ha', 'lmmse', 'cs', 'ssdd', 'mhc', 'gbtf', 'ri', 'mlri', 'ari', "aicc", "cdmcnn"]
DEFAULT_ALGS = ['bilinear', 'cs', 'ha', 'gunturk', 'lmmse', 'gbtf']

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--window-size", "-W", type=int, default=32)
    parser.add_argument("--logthreshold", "-t", type=float, default=3)
    parser.add_argument("--algorithms", "-a", nargs='+', choices=ALGS, default=DEFAULT_ALGS)
    args = parser.parse_args()

    img_path = args.input
    W = args.window_size
    algs = args.algorithms
    threshold = 10 ** (-args.logthreshold)
    detected_inconsistencies_diag, detected_inconsistencies_grid, detected_inconsistencies_pattern, detected_inconsistencies_alg, detected_inconsistencies_merged, detected_is_consistent_diag, detected_is_consistent_grid, detected_is_consistent_alg = detect_forgeries(
        img_path, algs, W, threshold)
    consistency_map_diag = detected_is_consistent_diag[:, :, None] * [0, 255, 0]  + detected_inconsistencies_diag[:, :, None] * [255, 0, 0]
    consistency_map_grid = detected_is_consistent_grid[:, :, None] * [0, 255, 0]  + detected_inconsistencies_grid[:, :, None] * [255, 0, 0]
    consistency_map_alg = detected_is_consistent_alg[:, :, None] * [0, 255, 0]  + detected_inconsistencies_alg[:, :, None] * [255, 0, 0]
    imageio.imsave("output.png", detected_inconsistencies_merged.astype(np.uint8)*255)
    imageio.imsave("consistency_map_diag.png", consistency_map_diag.astype(np.uint8))
    imageio.imsave("consistency_map_grid.png", consistency_map_grid.astype(np.uint8))
    imageio.imsave("consistency_map_alg.png", consistency_map_alg.astype(np.uint8))