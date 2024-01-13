import glob
import numpy as np
import pydicom
from numpy.fft import fft, fftfreq
from scipy import ndimage
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import Lookup_Data


def gauss(x, *p):
    a1, a2, a3, a4, a5 = p
    return a2 ** 2 * np.exp(-0.5 * ((x - a1) / a3) ** 2)


def pre_init():
    Lookup_Data.init_experiment_data()


def init(root, name, operation):
    params = None
    cor_x = None
    cor_y = None
    cor = None
    if operation == 'ESF':
        params = Lookup_Data.get_experiment_data(root, name, operation)

        cor_x = params.cor_x
        cor_y = params.cor_y
    elif operation == 'LSF' or operation == 'CT':
        params = Lookup_Data.get_experiment_data_ct(root, name)
        cor = params.cor
    slices = params.slices
    path = params.path
    factor = params.factor

    files = sorted(glob.glob(path + '/*.dcm'), key=len)
    data = []

    for file in files:
        data.append(pydicom.dcmread(file))

    slices = data[slices[0]:slices[1]]
    s = np.array([s.pixel_array for s in slices])

    if operation == 'ESF':
        return s, cor_x, cor_y, data, factor
    else:
        return s, cor, data, factor


def esf(root, name, operation):
    s, cor_y, cor_x, data, factor = init(root, name, operation)
    z_summed = np.zeros((cor_x[1] - cor_x[0], cor_y[1] - cor_y[0]))
    for i, sli in enumerate(s):
        sl = sli[cor_x[0]:cor_x[1], cor_y[0]:cor_y[1]]

        z_summed = z_summed - sl

    z_summed = z_summed / s.shape[0]

    esf_ = - np.sum(z_summed, axis=1) / z_summed.shape[1]

    lsf = np.diff(esf_)

    noise_ind = [0, 1, 2, -3, -2, -1]
    x = np.arange(len(lsf), dtype='float')
    x = x * data[0].PixelSpacing[0]

    lsf = linear_detrend(lsf, noise_ind)
    lsf = lsf - np.mean(lsf[noise_ind])
    p0 = [1., 1., 1., 1., 1.]
    function = gauss
    x_pred = np.arange(x[0], x[-1], 0.01 * data[0].PixelSpacing[0])
    p_opt, p_cov = curve_fit(function, x, lsf, p0=p0)
    print(p_opt)
    y_pred = function(x_pred, *p_opt)
    y_max = np.max(y_pred)
    y_pred = y_pred / y_max
    lsf = lsf / y_max

    x_max = x_pred[np.argmax(y_pred)]
    x = x - x_max
    x_pred = x_pred - x_max
    # plt.plot(x,LSF)
    # plt.plot(x_pred,y_pred)
    # plt.grid()
    # plt.show()
    return x, lsf, x_pred, y_pred


def zero_intersection(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1  # y!=0  y=mx+c
    x = -c / m
    return x


def calc_area_left(x, y):
    area = 0
    for i in range(1, len(y)):
        area += 0.5 * (y[i] - y[i - 1]) * (x[i] - x[i - 1])
        if i > 1:
            area += (x[i] - x[i - 1]) * y[i - 1]
    return area


def calc_area_right(x, y):
    area = 0
    for i in range(len(y) - 2, -1, -1):
        area += 0.5 * (y[i] - y[i + 1]) * (x[i + 1] - x[i])
        if i < len(y) - 2:
            area += (x[i + 1] - x[i]) * y[i + 1]
    return area


def linear_detrend(y, noise_ind):
    x = np.arange(0, len(y)).reshape((-1, 1))
    trend_y = y[noise_ind]
    trend_x = x[noise_ind]
    model = LinearRegression()
    model.fit(trend_x, trend_y)
    for i, val in enumerate(y):
        y[i] = y[i] - (model.coef_ * x[i] + model.intercept_)
    return y


def calc_missing_area(left_x, left_y, right_x, right_y, left, right):
    search_points = np.linspace(left, right, 10)
    search_points = search_points[1:-1]
    min_area = 100
    x_mid = 0
    for p in search_points:
        a_left_calc = calc_area_left(np.append(left_x, p), np.append(left_y, 1))
        a_right_calc = calc_area_right(np.append(p, right_x), np.append(1, right_y))
        if np.abs(a_left_calc - a_right_calc) < min_area:
            min_area = np.abs(a_left_calc - a_right_calc)
            x_mid = p
    return x_mid


def find_intersection_with_zero(y):
    size = len(y) // 2
    y_left = y[0:size]
    y_right = y[size:-1]
    first_index = 0
    second_index = 0
    for i in range(0, len(y_left) - 1):
        if y_left[i] < 0:
            first_index = i

    for j in range(len(y_right) - 1, 0, -1):
        if y_right[j] < 0:
            second_index = j + size
    if first_index >= second_index:
        first_index = 0
    if second_index <= first_index:
        second_index = size - 1

    return first_index, second_index


def find_center(y):
    highest = np.argmax(y)
    left = highest - 1
    right = highest + 1
    first_index, second_index = find_intersection_with_zero(y)

    zero_left = zero_intersection(first_index, y[first_index], first_index + 1, y[first_index + 1])
    zero_right = zero_intersection(second_index - 1, y[second_index - 1], second_index, y[second_index])

    left_x = np.append(np.array([zero_left]), np.arange(first_index + 1, highest))
    left_y = np.append(np.array(0), y[first_index + 1:highest])

    right_x = np.append(np.arange(highest + 1, second_index), np.array([zero_right]))
    right_y = np.append(y[highest + 1:second_index], np.array(0))

    return calc_missing_area(left_x, left_y, right_x, right_y, left, right)


def do_lsf_radial(root, name, operation):
    s, cor, data, factor = init(root, name, operation)
    roi = 10
    s = s - 180.0
    z_summed = np.zeros((2 * roi, 2 * roi))

    for i, sli in enumerate(s):
        sl = sli[cor[1] - roi:cor[1] + roi, cor[0] - roi:cor[0] + roi]
        z_summed = z_summed - sl
    z_summed = z_summed / s.shape[0]

    angles = np.arange(0, 180, 20)
    bins = 20
    rebin_x = np.linspace(-5, 5, bins)  # -15 bis 19
    re_x = np.zeros(bins)
    rebin_y = np.zeros(bins)
    bin_counter = np.zeros(bins)

    for angle in angles:
        img_rot = ndimage.rotate(z_summed, angle, reshape=False, mode='reflect')
        img_rot = -factor * img_rot

        noise_ind = [0, 1, 2, 3, -4, -3, -2, -1]
        line_int = np.sum(img_rot, axis=0) / img_rot.shape[0]
        line_int = linear_detrend(line_int, noise_ind)
        line_int = line_int - np.mean(line_int[noise_ind])
        line_int = line_int / np.max(line_int)

        mid = find_center(line_int)

        x = np.arange(len(line_int), dtype='float')
        x = x - mid
        argmax = np.argmax(line_int)

        x[argmax] = 0
        x = x * data[0].PixelSpacing[0]

        bin_places = np.digitize(x, rebin_x)

        for i, bin_place in enumerate(bin_places):
            if bin_place != bins:
                re_x[bin_place] += x[i]
                rebin_y[bin_place] += line_int[i]
                bin_counter[bin_place] += 1

    for i, count in enumerate(bin_counter):
        if count != 0:
            rebin_y[i] /= count
            re_x[i] /= count
    rebin_x = re_x

    to_delete = []
    for i in range(len(rebin_y)):
        if np.abs(rebin_y[i]) < 0.0001:
            to_delete.append(i)

    rebin_y = np.delete(rebin_y, to_delete)
    rebin_x = np.delete(rebin_x, to_delete)

    noise_ind = [0, 1, 2, 3, -4, -3, -2, -1]
    rebin_y = rebin_y - np.mean(rebin_y[noise_ind])
    rebin_y = rebin_y / np.max(rebin_y)

    rebin_x = rebin_x - rebin_x[np.argmax(rebin_y)]
    p0 = [1, 1., 1., 1., 1.]

    function = gauss
    pos_max = np.argmax(rebin_y)
    x_pred = np.arange(-pos_max, len(rebin_y) - pos_max, 0.01) * data[0].PixelSpacing[0]
    p_opt, p_cov = curve_fit(function, rebin_x, rebin_y, p0=p0)
    print(p_opt)
    print(p_cov)
    y_pred = function(x_pred, *p_opt)
    y_pred = y_pred / np.max(y_pred)

    spacing = data[0].PixelSpacing[0]
    x_max = x_pred[np.argmax(y_pred)]
    x_pred = x_pred - x_max
    return x_pred, y_pred, rebin_x, rebin_y, spacing


def do_mtf_org(x, y, spacing):
    if len(x) % 2 != 0:
        y = y[:-1]
        x = x[:-1]
    line = np.zeros(100000)
    line[50000 - int(len(x) / 2):50000 + int(len(x) / 2)] = y

    xf1, y1 = do_mtf(line, spacing)

    return xf1 * 10, y1


def do_mtf(line, spacing):
    yf = fft(line)
    x_f = np.arange(len(line)) * spacing
    n = len(x_f)
    xf1 = fftfreq(n, spacing)[:n // 2]
    y = np.abs(yf[0:n // 2])
    y1 = y / y[0]
    return xf1, y1


def do_mtf_fit(x, y, spacing):
    spacing = spacing * 0.01
    line = np.zeros(10000)
    line[5000 - int(len(x) / 2):5000 + int(len(x) / 2)] = y
    xf1, y1 = do_mtf(line, spacing)

    return xf1[0:100] * 10, y1[0:100]


def fw_hm_org(x, y):
    y = y - 0.5
    first_index, second_index = find_intersection_with_zero(y)
    x1 = x[first_index]
    x2 = x[first_index + 1]
    x3 = x[second_index - 1]
    x4 = x[second_index]
    x_left = zero_intersection(x1, y[first_index], x2, y[first_index + 1])
    x_right = zero_intersection(x3, y[second_index - 1], x4, y[second_index])
    fw_hm = np.round(np.abs(x_left) + x_right, 3)
    return fw_hm


def fw_hm_fit(x, y):
    y1 = np.argmin(np.abs(y[:int(len(y) / 2)] - 0.5))
    y2 = np.argmin(np.abs(y[int(len(y) / 2):-1] - 0.5))
    fw_hm = np.round(np.abs(x[y1]) + x[y2 + int(len(y) / 2)], 3)
    return fw_hm


def mtf_val_fit(x, y):
    x_50 = np.round(x[np.argmin(np.abs(y - 0.5))], 2)
    x_10 = np.round(x[np.argmin(np.abs(y - 0.1))], 2)
    return x_50, x_10


def mtf_val_org(x, y):
    index_50 = 0
    index_10 = 0
    y_50 = y - 0.5
    y_10 = y - 0.1
    for i in range(len(y)):
        if y_50[i] < 0:
            index_50 = i
            break
    for i in range(len(y)):
        if y_10[i] < 0:
            index_10 = i
            break
    x1 = x[index_50]
    x2 = x[index_50 + 1]
    x_50 = np.round(zero_intersection(x1, y_50[index_50], x2, y_50[index_50 + 1]), 2)
    x1 = x[index_10]
    x2 = x[index_10 + 1]
    x_10 = np.round(zero_intersection(x1, y_10[index_10], x2, y_10[index_10 + 1]), 2)
    return x_50, x_10
