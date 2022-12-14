#!/usr/bin/env python
import sys
import imageio
import numpy as np

from analysis import detect_forgeries

ALGS = ['bilinear', 'gunturk', 'ha', 'lmmse', 'cs', 'ssdd', 'mhc', 'gbtf', 'ri', 'mlri', 'ari', "aicc", "cdmcnn"]
DEFAULT_ALGS = ['bilinear', 'cs', 'ha', 'gunturk', 'lmmse', 'gbtf']

if __name__ == "__main__":
    import argparse

    class ArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            with open("demo_failure.txt", "w") as f:
                f.write(message)
            self.exit(0, '%s: error: %s\n' % (self.prog, message))
    parser = ArgumentParser(description="Detection of image forgeries by use of the double demosaicing method.")
    parser.add_argument("input", type=str, help="path to the input image.")
    parser.add_argument("--window-size", "-W", type=int, default=32, help="Size of the windows in which to perform detections. Larger windows will lead to more confident results but might cause smaller forgeries to be undetected. Increase the size if the method does not perform well locally, decrease it to find smaller forgeries if the method performs well locally at the selected size. Window size should be even.")
    parser.add_argument("--logthreshold", "-t", type=float, default=3, help="negative base-10 logarithm of the tolerated number of false alarms (NFA). A value of k means that under the background hypothesis, one can statistically expect one false detection every 10^k images.")
    parser.add_argument("--algorithms", "-a", nargs='+', choices=ALGS, default=DEFAULT_ALGS, help="Algorithms to use in the detection.")
    args = parser.parse_args()
    img_path = args.input
    W = args.window_size
    Wn = W - W % 2
    if W != Wn:
        print(f"Window size must be even. Modifying it from {W} to {Wn}.")
        Wn = W
    algs = args.algorithms
    if len(algs) == 0:
        print("An empty algorithms list was provided. Using the default selection instead.")
        algs = DEFAULT_ALGS

    threshold = 10 ** (-args.logthreshold)
    detected_inconsistencies_diag, detected_inconsistencies_grid, detected_inconsistencies_pattern, detected_inconsistencies_alg, detected_inconsistencies_merged, detected_is_consistent_diag, detected_is_consistent_grid, detected_is_consistent_alg = detect_forgeries(
        img_path, algs, W, threshold)
    consistency_map_diag = detected_is_consistent_diag[:, :, None] * [0, 255, 0] + detected_inconsistencies_diag[:, :,
                                                                                   None] * [255, 0, 0]
    consistency_map_grid = detected_is_consistent_grid[:, :, None] * [0, 255, 0] + detected_inconsistencies_grid[:, :,
                                                                                   None] * [255, 0, 0]
    consistency_map_alg = detected_is_consistent_alg[:, :, None] * [0, 255, 0] + detected_inconsistencies_alg[:, :,
                                                                                 None] * [255, 0, 0]
    imageio.imsave("output.png", detected_inconsistencies_merged.astype(np.uint8) * 255)
    imageio.imsave("consistency_map_diag.png", consistency_map_diag.astype(np.uint8))
    imageio.imsave("consistency_map_grid.png", consistency_map_grid.astype(np.uint8))
    imageio.imsave("consistency_map_alg.png", consistency_map_alg.astype(np.uint8))
