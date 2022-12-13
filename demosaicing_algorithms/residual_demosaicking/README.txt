-------------------------------------------------------------------

A Mathematical Analysis and Implementation of Residual Interpolation Demosaicking Algorithms

-------------------------------------------------------------------

Copyright (c) 2020 Inner Mongolia University and ENS Paris-Saclay. 
All rights reserved.

Authors:                     Qiyu Jin
                             Yu Guo 
                             Gabriele Facciolo 
                             Jean-Michel Morel

-------------------------------------------------------------------

Algorithm 1 and 2 are implemented in file 'demosaic_HA.py'
Algorithm 3 is implemented in file 'RIgreen_interpolation.py' and 'RIHaResidual.py'
Algorithm 4 is implemented in file 'RIred_interpolation.py'
Algorithm 11 is implemented in file 'RIguidedfilter3gf.py'
Algorithm 5 is implemented in file 'RIgreen_interpolation.py' and 'RIGuidefilterResidual.py'
Algorithm 6 is implemented in file 'RIred_interpolation.py'
Algorithm 7 and 8 is implemented in file 'ARIgreen_interpolation.py'
Algorithm 9 is implemented in file 'ARIred_blue_interpolation_first.py'
Algorithm 10 is implemented in file 'ARIred_blue_interpolation_second.py'

-------------------------------------------------------------------
 Requirements
-------------------------------------------------------------------

Required python:
python3

Required python package:
numpy, opencv-python, scikit-image

-------------------------------------------------------------------
 Contents
-------------------------------------------------------------------

This project implements several traditional demosaicing algorithms in one python program.
In this project, we can use the algorithm('HA', 'GBTF', 'RI', 'MLRI', 'WMLRI', 'ARI') to demosaicing through parameter selection

# run.py [-h] [--input INPUT] [--pattern PATTERN] [--Algorithm ALGORITHM] [--output OUTPUT] [--noise_sigma NOISE_SIGMA] [--mosaic MOSAIC] [--sigma SIGMA]

--INPUT: "test input, uses the default test input provided if no argument"

--PATTERN: ""bayer pattern"". The default choice is 'grbg'
		'grbg', 'rggb', 'gbrg', 'bggr'

--ALGORITHM: "Demosaicing Algorithm". The default choice is 'GBTF'
 		'HA', 'GBTF', 'RI', 'MLRI', 'WMLRI', 'ARI'

--OUTPUT: "output image address"

--NOISE_SIGMA: "standard deviation of the simulated additive Gaussian noise"

--MOSAIC: "noisy mosaic image input to the algorithm"

--SIGMA: "sigma regularization parameter"



-------------------------------------------------------------------
Usage example
-------------------------------------------------------------------

$ python run.py --input Sans_bruit_13.PNG --pattern grbg --Algorithm GBTF --output test_GBTF.png 

-------------------------------------------------------------------
 Feedback
-------------------------------------------------------------------

If you have any comment, suggestion, or question, please do contact
  Qiyu Jin  at  qyjin2015@aliyun.com
  Yu Guo    at  yuguomath@aliyun.com
