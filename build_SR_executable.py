import os
import PyInstaller.__main__
import sys
sys.setrecursionlimit(50000)

PyInstaller.__main__.run([
    '--name=%s' %'ScaredyRat',
    '--onefile',
    '--hidden-import=%s' %'tkinter,scipy,numpy,pandas,openpyxl,matplotlib,seaborn,PySimpleGUI',
    '--additional-hooks-dir=%s' %'.',
    '--clean',
    '--upx-dir=%s\,%''./UPX/upx-3.96-win64',
    '--paths=%s' %'./src',
    './SR_GUI.py',
    './src/sr_compiled.py',
    './src/sr_functions.py',
    './src/sr_individual.py',
    './src/sr_settings.py'
])
