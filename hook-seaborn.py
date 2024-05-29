# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:08:20 2024

@author: jraqu
"""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('seaborn')