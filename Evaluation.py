from pathlib import Path

import matplotlib
import numpy as np
import pydicom
import pandas as pd
import pymedphys

import glob
import seaborn as sns
import os
import re
import PatientEvaluation

# gauss, rect, noise, both, gauss_noise
mods = 'rect'

folder_selected = None
if mods == 'rect':
    folder_selected = Path("../data/rect")
if mods == 'gauss':
    folder_selected = Path(r"../data/gauss")
if mods == 'noise':
    folder_selected = Path("../data/noise")
if mods == 'gauss_noise':
    folder_selected = Path("../data/gauss_noise")

paths = [f for f in folder_selected.iterdir() if f.is_dir()]
patient_names = [p.name for p in paths]
print(patient_names)
values = input("Enter indices of patient to evaluate comma seperated:\n")
values = values.split(',')
values = [x.zfill(2) for x in values]
p = []
pat = []
for path, patient in zip(paths, patient_names):
    if re.search('\d{2}', patient)[0] in values:
        p.append(path)
        pat.append(patient)
paths = p
patient_names = pat
for p, pat in zip(paths, patient_names):
    PatientEvaluation.progress(str(p), pat, mods)
