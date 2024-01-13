import os
from enum import Enum
import Calculation
import matplotlib.pyplot as plt
import pandas as pd


# possible operations: LSF, ESF, CT

class Names(Enum):
    FDG_151 = 'FDG_151'
    FDG_152 = 'FDG_152'
    FDG_201 = 'FDG_201'
    FDG_202 = 'FDG_202'
    FDG_301 = 'FDG_301'
    FDG_302 = 'FDG_302'
    FDG_PET_15 = 'FDG_PET_15'
    FDG_PET_20 = 'FDG_PET_20'
    FDG_61 = 'FDG_61'
    FDG_62 = 'FDG_62'
    FDG_11 = 'FDG_11'
    FDG_12 = 'FDG_12'
    GA_151 = 'GA_151'
    GA_152 = 'GA_152'
    GA_201 = 'GA_201'
    GA_202 = 'GA_202'
    ub_039 = 'ub_039'
    b_039 = 'b_039'
    ub_069 = 'ub_068'
    ub_077 = 'ub_077'
    b_097 = 'b_097'
    b_117 = 'b_117'


names = [e.value for e in Names if 'FDG' not in e.value and 'GA' not in e.value]
Calculation.pre_init()
operation = 'CT'
root = '../../Experimente'
save = '../../Experimente/images'
if not os.path.exists(save):
    # Create a new directory because it does not exist
    os.makedirs(save)
FWHMs_fit = []
FWHMs_org = []
MTF50s_org = []
MTF50s_fit = []
MTF10s_org = []
MTF10s_fit = []
for name in names:
    plt.figure()
    x_pred, y_pred, rebin_x, rebin_y, spacing = Calculation.do_lsf_radial(root, name, operation)

    plt.grid(b=True)
    plt.legend()
    save_path = save + '/' + name
    FWHM_fit = Calculation.fw_hm_fit(x_pred, y_pred)
    FWHMs_fit.append(FWHM_fit)
    FWHM_org = Calculation.fw_hm_org(rebin_x, rebin_y)
    FWHMs_org.append(FWHM_org)

    plt.savefig(save_path)
    xf1_fit, yf_fit = Calculation.do_mtf_fit(x_pred, y_pred, spacing)
    x50, x10 = Calculation.mtf_val_fit(xf1_fit, yf_fit)
    MTF50s_fit.append(x50)
    MTF10s_fit.append(x10)
    print(str(x50) + ' ,' + str(x10))
    # plt.plot(xf1_fit, yf_fit)
    plt.grid()
    # plt.show()
    xf1_org, yf_org = Calculation.do_mtf_org(rebin_x, rebin_y, spacing)
    x50, x10 = Calculation.mtf_val_org(xf1_org, yf_org)
    MTF50s_org.append(x50)
    MTF10s_org.append(x10)
    print(str(x50) + ' ,' + str(x10))
    # plt.plot(xf1_org, yf_org)
    plt.grid()
    # plt.show()
d = {
    "MTF50_org": pd.Series(MTF50s_org, index=names),
    'MTF50_fit': pd.Series(MTF50s_fit, index=names),
    "MTF10_org": pd.Series(MTF10s_org, index=names),
    "MTF10_fit": pd.Series(MTF10s_fit, index=names)
}

df = pd.DataFrame(d)
file_name = root + '/MTF_val_CT.csv'
df.to_csv(file_name)

names2 = [e.value for e in Names if 'FDG' in e.value]

Calculation.pre_init()
operation = 'ESF'
root = '../../Experimente'
save = '../../Experimente/images'
if not os.path.exists(save):
    # Create a new directory because it does not exist
    os.makedirs(save)
for name in names2:
    x, LSF, x_pred, y_pred = Calculation.esf(root, name, operation)
    name = save + '/' + name
    FWHM_fit = Calculation.fw_hm_fit(x_pred, y_pred)
    FWHMs_fit.append(FWHM_fit)
    print(name)
    print(FWHM_fit)
    FWHM_org = Calculation.fw_hm_org(x, LSF)
    FWHMs_org.append(FWHM_org)
    print(FWHM_org)
names = names + names2

d = {
    "FWHM_fit": pd.Series(FWHMs_fit, index=names),
    "FWHM_org": pd.Series(FWHMs_org, index=names),
}

df = pd.DataFrame(d)
file_name = root + '/FWHM_all.csv'
df.to_csv(file_name)
