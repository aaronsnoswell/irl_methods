# -*- coding: utf-8 -*-
"""__init__.py for the irl_methods.thirdparty module

This submodule contains code written by third parties that doesn't exist as
a stand-alone python package anywhere.

Copyright 2018 Aaron Snoswell
"""

from .robsvm import robust_svm_fit

__all__ = [
    "robust_svm_fit"
]
