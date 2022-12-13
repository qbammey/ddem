from pathlib import Path

import numpy as np
from skimage.util import view_as_blocks
import imageio

from demosaic import demosaic

def get_maps(img_path, algs):
    img_path = Path(img_path)
    img = imageio.imread(img_path)[:, :, :3] / 255
    Y, X, _ = img.shape
    residuals = np.zeros((len(algs), 4, Y, X, 3))
    for grid in range(4):
        for i_alg, alg in enumerate(algs):
            residuals[i_alg, grid] = demosaic(img_path, grid, alg) / 255 - img
    Y -= Y % 2
    X -= X % 2
    residuals = np.mean(np.square(residuals[:, :, :Y, :X]), axis=-1)
    residuals = view_as_blocks(residuals, (len(algs), 4, 2, 2)).squeeze()
    residuals = residuals.mean(axis=(-1, -2))
    best_grid = np.argmin(np.min(residuals, axis=2), axis=2)
    best_alg = np.argmin(np.min(residuals, axis=3), axis=2)
    
