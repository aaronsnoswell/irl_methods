# -*- coding: utf-8 -*-
"""__init__.py for the irl_methods module

Copyright 2018 Aaron Snoswell
"""

import irl_methods.utils
import irl_methods.thirdparty
import irl_methods.mdp

from .linear_programming import (
    linear_programming,
    large_linear_programming,
    trajectory_linear_programming
)
from .projection import projection, find_mixing_weights
from .maximum_margin import maximum_margin
from .maximum_entropy import maximum_entropy
from .deep_maximum_entropy import deep_maximum_entropy


# OpenBLAS warning message
openblas_warn_msg = """It looks like you might be using Numpy with OpenBLAS \
on Windows. If your OpenBLAS version is 0.2.0, there is a bug that causes \
np.linalg.inv() to deadlock for matrices larger than 24x24. As a workaround, \
try executing `set OPENBLAS_NUM_THREADS=1` at the command line before \
exexuting any IRL method. Please see \
https://github.com/numpy/numpy/issues/11041#issuecomment-386521546 for more \
information"""

# Test for OpenBLAS and Windows
import os
import numpy as np
if np.__config__.get_info("openblas_info") and os.name == "nt":
    # Check if OPENBLAS_NUM_THREADS=1 already
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        import warnings
        warnings.warn(openblas_warn_msg, RuntimeWarning)


__all__ = [
    "utils",
    "thirdparty",
    "mdp",

    "linear_programming",
    "projection",
    "maximum_margin",
    "maximum_entropy",
    "deep_maximum_entropy"
]
