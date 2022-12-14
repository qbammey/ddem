#!/usr/bin/env python
import sys
import os.path
import subprocess
import argparse
from pathlib import Path

import numpy as np
import imageio

sys.path.append(os.path.join(os.path.dirname(__file__), "demosaicing_algorithms/residual_demosaicking"))
from mosaic_bayer import mosaic_bayer
from demosaic_RI import demosaic_RI
from demosaic_ARI import demosaic_ARI
sys.path.append(os.path.join(os.path.dirname(__file__), "demosaicing_algorithms/cdm_cnn"))
from cdmcnn import main as cdmcnn

_pattern_dict = {
    "RGGB": 0,
    "rggb": 0,
    0: 0,
    "BGGR": 1,
    "bggr": 1,
    1: 1,
    "GRBG": 2,
    "grbg": 2,
    2: 2,
    "GBRG": 3,
    "gbrg": 3,
    3: 3,
}


def _get_pattern_flag(method, pattern):
    pattern_id = _pattern_dict[pattern]
    if method in ['bilinear', 'gunturk', 'ha', 'lmmse', 'cs', 'ssdd', 'mhc']:
        pattern_flag = ["RGGB", "BGGR", "GRBG", "GBRG"][pattern_id]
    elif method in ['gbtf', 'ri', 'mlri', 'ari']:
        pattern_flag = ['rggb', 'bggr', 'grbg', 'gbrg'][pattern_id]
    elif method == "aicc":
        pattern_flag = ['0 0', '1 1', '0 1', '1 0'][pattern_id]
    elif method == "cdmcnn":
        oy = 0 if pattern_id in [0, 3] else 1
        ox = 0 if pattern_id in [0, 2] else 1
        pattern_flag = [ox, oy]
    else:
        raise ValueError(f"method {method} unknown")
    return pattern_flag


def demosaic(img, method, pattern, **kwargs):
    img = Path(img)
    pattern = _pattern_dict[pattern]
    out_path = Path("tmp") / f"{method}_{pattern}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    demosaicing_path = Path(__file__).parent / "demosaicing_algorithms"
    demosaicing_path = demosaicing_path.absolute()
    pattern = _get_pattern_flag(method, pattern)
    if method in ['bilinear', 'gunturk', 'ha', 'lmmse', 'cs', 'mhc']:
        path = None
        if method in ['bilinear', 'gunturk', 'ha']:
            path = demosaicing_path / "gunturk" / f"dm{method}"
        elif method == 'lmmse':
            path = demosaicing_path / "lmmse" / "dmzhangwu"
        elif method == 'cs':
            path = demosaicing_path / "cs" / "dmcswl1"
        elif method == 'mhc':
            path = demosaicing_path / "mhc" / f"dmmalvar"
        path = path.as_posix()
        subprocess.run([str(path), "-p", str(pattern), str(img), str(out_path)], stdout=subprocess.DEVNULL)
        out = imageio.imread(out_path)
    elif method in ['gbtf', 'ri', 'mlri', 'ari']:
        img = imageio.imread(img)[:, :, :3] / 255
        if img.ndim == 2:
            img = img[:, :, None].repeat(3, axis=-1)
        mos, _ = mosaic_bayer(img, pattern)
        alg = method.upper()
        if alg == "ARI":
            out = demosaic_ARI(mos, pattern)
        else:
            out = demosaic_RI(mos, pattern, 0, alg)
        #out = demosaic_ri(img, pattern, method.upper())
        out = np.round(out*255).astype(np.uint8)
    elif method == "ssdd":
        path = demosaicing_path / "ssdd" / "demosaickingIpol"
        path = path.as_posix()
        img_tiff = img.with_suffix(".tiff")
        subprocess.run(["convert", img.as_posix(), img_tiff.as_posix()])
        out_tiff = out_path.with_suffix(".tiff")
        subprocess.run([path, img_tiff.as_posix(), out_tiff.as_posix(), str(pattern)])
        out = imageio.imread(out_tiff)
    elif method == "aicc":
        path = demosaicing_path / "aicc" / "demosaicking_ipol"
        subprocess.run([path.as_posix(), img.as_posix(), out_path.with_stem("_").as_posix(), out_path.as_posix(), pattern, "0"])
        out = imageio.imread(out_path)
    elif method == "cdmcnn":
        args = argparse.Namespace()
        args.input = img.as_posix()
        args.net_path = None
        args.offset_x = pattern[0]
        args.offset_y = pattern[1]
        args.gpu = False
        args.linear_input = False
        out, _ = cdmcnn(args)

    else:
        raise ValueError(f"demosaicing method {method} is not known or supported.")
    return out[:, :, :3]

"""
def demosaic(img, method, pattern, **kwargs):
    img = Path(img)
    pattern = _pattern_dict[pattern]
    out = Path("tmp") / f"{method}_{pattern}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    commands = _get_demosaic_cmd(img, out, method, pattern, **kwargs)
    for cmd in commands:
        subprocess.run(cmd)
    out = imageio.imread(out)
    return out
"""

if __name__ == "__main__":
    methods = ['bilinear', 'gunturk', 'ha', 'lmmse', 'cs', 'ssdd', 'gbtf', 'ri', 'mlri', 'ari', "aicc", "cdmcnn"]
    methods = ["mhc"]
    patterns = ["rggb", "bggr", "grbg", "gbrg"]
    for method_ in methods:
        for pattern_ in patterns:
            out = demosaic("input_0.png", method_, pattern_)
            imageio.imsave(f"out_{method_}_{pattern_}.png", out)