# Demosaicing to Detect Demosaicing and Image Forgeries

This is the official repository for the source code of the article *Demosaicing to Detect Demosaicing and Image Forgeries*, by Quentin Bammey, Rafael Grompone von Gioi, and Jean-Michel Morel.

An online demo of the algorithm is available on [IPOL](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000388). This link will change in the future, in the meantime please link to this page rather than directly to the demo.

# Usage
A Docker image is provided in `.ipol/Docker`. 

While inside the Docker image, the code can be run as following:

usage: `main.py [-h] [--window-size WINDOW_SIZE] [--logthreshold LOGTHRESHOLD]
[--algorithms {bilinear,gunturk,ha,lmmse,cs,ssdd,mhc,gbtf,ri,mlri,ari,aicc,cdmcnn}]
input`

positional argument:

`input                 path to the input image.`

options:
* `-h, --help`:            show this help message and exit
* `--window-size WINDOW_SIZE, -W WINDOW_SIZE`: Size of the windows in which to perform detections. Larger windows will lead to more confident
results but might cause smaller forgeries to be undetected. Increase the size if the method
does not perform well locally, decrease it to find smaller forgeries if the method performs
well locally at the selected size. Window size should be even.
* `--logthreshold LOGTHRESHOLD, -t LOGTHRESHOLD`: negative base-10 logarithm of the tolerated number of false alarms (NFA). A value of k means
that under the background hypothesis, one can statistically expect one false detection every
10^k images.
* `--algorithms {bilinear,gunturk,ha,lmmse,cs,ssdd,mhc,gbtf,ri,mlri,ari,aicc,cdmcnn}`: Algorithms to use in the detection.

As an alternative to Docker, it is also possible to manually install the packages in `.ipol/packages.txt` (Debian packages for compilation, their names may vary depending on your distribution or might already be installed), and the python libraries in `requirements.txt`.

Demosaicing algorithms in the `demosaicing_algorithms` folder should then be manually compiled except for `demosaicing_algorithms/cdm_cnn` and `demosaicing_algorithms/residual_demosaicing`. See the individual folders for instructions about compilation.

# LICENCE

The files in the folder "demosaicing_algorithms" come from other sources and are licensed separately, as detailed below and in the respective folders. All the other files are copyright Quentin Bammey, and are distributed under the GPL 3.0 or later licence. The following applies for the files that are not in the demosaicing_algorithms folder.

Copyright (C) 2022 Quentin Bammey

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.



