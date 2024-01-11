import glob
import numpy as np
import pydicom
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import Lookup_Data

def gauss(x, *p):
    a1, a2, a3, a4, a5 = p
    return a2 ** 2 * np.exp(-0.5 * ((x - a1) / a3) ** 2)


def linear_detrending(y):
    detrend_indizes = [0, 1, 2, -3, -2, -1]
    x = np.arange(0, len(y)).reshape((-1, 1))
    trend_y = y[detrend_indizes]
    trend_x = x[detrend_indizes]
    model = LinearRegression()
    model.fit(trend_x, trend_y)

    for i, val in enumerate(y):
        y[i] = y[i] - (model.coef_ * x[i] + model.intercept_)
    return y


def init():
    Lookup_Data.init_experiment_data()

def init(root,name):

    slices = params['slice']
    cor_x = params['cor_y']
    cor_y = params['cor_x']
    files = sorted(glob.glob(path + '/*.dcm'), key=len)
    data = []

    for file in files:
        data.append(pydicom.dcmread(file))

    slices = data[slices[0]:slices[1]]
    s = np.array([s.pixel_array for s in slices])
    z_summed = np.zeros((cor_x[1] - cor_x[0], cor_y[1] - cor_y[0]))
    for i, slice in enumerate(s):
        sl = slice[cor_x[0]:cor_x[1], cor_y[0]:cor_y[1]]

        z_summed = z_summed - sl
    z_summed = z_summed / s.shape[0]
    # plt.imshow(z_summed,cmap='gray')
    # plt.show()
    ESF = - np.sum(z_summed, axis=1) / z_summed.shape[1]
    # plt.plot(ESF)
    # plt.show()
    LSF = np.diff(ESF)
    # plt.plot(LSF)
    # plt.show()
    noise_ind = [0, 1, 2, -3, -2, -1]
    x = np.arange(len(LSF), dtype='float')
    x = x * data[0].PixelSpacing[0]
    # plt.plot(x,line_int, marker='o')
    # plt.grid()
    # plt.show()
    LSF = linear_detrending(LSF)
    LSF = LSF - np.mean(LSF[noise_ind])
    p0 = [1., 1., 1., 1., 1.]
    function = func
    x_pred = np.arange(x[0], x[-1], 0.01 * data[0].PixelSpacing[0])
    popt, pcov = curve_fit(function, x, LSF, p0=p0)
    print(popt)
    y_pred = function(x_pred, *popt)
    y_max = np.max(y_pred)
    y_pred = y_pred / y_max
    LSF = LSF / y_max
    # plt.plot(x,LSF)
    # plt.plot(x_pred,y_pred)
    # plt.show()
    x_max = x_pred[np.argmax(y_pred)]
    x = x - x_max
    x_pred = x_pred - x_max
    # plt.plot(x,LSF)
    # plt.plot(x_pred,y_pred)
    # plt.grid()
    # plt.show()
    return x, LSF, x_pred, y_pred
