from pathlib import Path
import natsort
import matplotlib
import numpy as np
import dash
from dash import html
import imageio
from dash_slicer import VolumeSlicer
import PIL.Image
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import imageio
import pydicom
import pandas as pd
import pymedphys
from tkinter import filedialog
from tkinter import *
import cv2
import glob
import seaborn as sns
import os
import re
from scipy.optimize import curve_fit
import dicts


def init():
    return dic_diff, dic_gamma, dic_stats


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def get_cts(id):
    path = r"../data/output_data/png_org" + "/" + id
    ct_paths = glob.glob(path + '/*.png')
    ct_paths = natsort.natsorted(ct_paths, reverse=True)
    cts = []
    for i, ct in enumerate(ct_paths):
        cts.append(np.asarray(PIL.Image.open(ct)))
    cts = np.dstack(cts)
    return cts


def make_overlay_gamma(id, cts, roi, gammas, modus, name):
    data_path_out = "../data/output_data/gamma_overlay/"
    data_path_out = data_path_out + modus + '/' + id + '/' + name
    Path(data_path_out).mkdir(parents=True, exist_ok=True)
    roi = roi / 255
    roi = roi > 0.0
    roi = roi.astype(np.uint8)
    ct_corner = np.asarray(ct_corners[id])
    grid_corner = np.asarray(grid_corners[id])
    corner_indexes = np.round(np.abs(ct_corner - grid_corner) * 10).astype(np.int)
    gammas = np.transpose(gammas, axes=[2, 1, 0])
    roi = np.transpose(roi, axes=[2, 1, 0])
    gammas = gammas.astype(cts.dtype)
    cts = cts[corner_indexes[1]:corner_indexes[1] + gammas.shape[1],
          corner_indexes[0]:corner_indexes[0] + gammas.shape[0], corner_indexes[2]:corner_indexes[2] + gammas.shape[2]]
    gammas = np.transpose(gammas, axes=[1, 0, 2])
    roi = np.transpose(roi, axes=[1, 0, 2])
    out_img = []
    for i in range(cts.shape[2]):
        # transversal
        out_path = data_path_out + '/transversal'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        heatmap_img = cv2.applyColorMap(gammas[:, :, i], cv2.COLORMAP_JET)
        ct = cv2.merge((cts[:, :, i], cts[:, :, i], cts[:, :, i]))
        added_image = cv2.addWeighted(ct, 0.1, heatmap_img, 0.9, 0)
        roi_img = cv2.bitwise_and(added_image, added_image, mask=gammas[:, :, i])
        roi_img[gammas[:, :, i] == 0, ...] = ct[gammas[:, :, i] == 0, ...]
        contours, hierarchy = cv2.findContours(roi[:, :, i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi_img, contours, -1, (0, 255, 0), 1)
        out_path = out_path + '/' + str(i) + '.png'
        scale_percent = 500
        # calculate the 50 percent of original dimensions
        width = int(roi_img.shape[1] * scale_percent / 100)
        height = int(roi_img.shape[0] * scale_percent / 100)
        dsize = (width, height)
        output = cv2.resize(roi_img, dsize)
        cv2.imwrite(out_path, output)
        out_img.append(output)
    for i in range(cts.shape[1]):
        # sagital
        out_path = data_path_out + '/sagital'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        heatmap_img = cv2.applyColorMap(gammas[:, i, :], cv2.COLORMAP_JET)
        ct = cv2.merge((cts[:, i, :], cts[:, i, :], cts[:, i, :]))
        added_image = cv2.addWeighted(ct, 0.1, heatmap_img, 0.9, 0)
        roi_img = cv2.bitwise_and(added_image, added_image, mask=gammas[:, i, :])
        roi_img[gammas[:, i, :] == 0, ...] = ct[gammas[:, i, :] == 0, ...]
        contours, hierarchy = cv2.findContours(roi[:, i, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi_img, contours, -1, (0, 255, 0), 1)
        out_path = out_path + '/' + str(i) + '.png'
        scale_percent = 500
        # calculate the 50 percent of original dimensions
        width = int(roi_img.shape[1] * scale_percent / 100)
        height = int(roi_img.shape[0] * scale_percent / 100)
        dsize = (width, height)
        output = cv2.resize(roi_img, dsize)
        output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(out_path, output)
        out_img.append(output)

    for i in range(cts.shape[0]):
        # coronal
        out_path = data_path_out + '/coronal'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        heatmap_img = cv2.applyColorMap(gammas[i, :, :], cv2.COLORMAP_JET)
        ct = cv2.merge((cts[i, :, :], cts[i, :, :], cts[i, :, :]))
        added_image = cv2.addWeighted(ct, 0.1, heatmap_img, 0.9, 0)
        roi_img = cv2.bitwise_and(added_image, added_image, mask=gammas[i, :, :])
        roi_img[gammas[i, :, :] == 0, ...] = ct[gammas[i, :, :] == 0, ...]
        contours, hierarchy = cv2.findContours(roi[i, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi_img, contours, -1, (0, 255, 0), 1)
        out_path = out_path + '/' + str(i) + '.png'
        scale_percent = 500
        # calculate the 50 percent of original dimensions
        width = int(roi_img.shape[1] * scale_percent / 100)
        height = int(roi_img.shape[0] * scale_percent / 100)
        dsize = (width, height)
        output = cv2.resize(roi_img, dsize)
        output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(out_path, output)
        out_img.append(output)


def path_extraction(path, paths, folder_selected):
    path_evaluations = folder_selected + '/doses/*.npy'
    path_evaluations = glob.glob(path_evaluations)
    paths['path_evaluation'] = path_evaluations
    path_original = list(filter(lambda x: 'CT 1' in x, path_evaluations))[0]
    paths['path_original'] = path_original
    print('original file is ' + str(path_original))
    path_evaluations.remove(path_original)
    path_gamma = folder_selected + '/gamma/*.npy'
    path_gamma = glob.glob(path_gamma)
    # path_gamma = list(filter(lambda x: 'gamma' in x, gamma_folder))
    paths['path_gamma'] = path_gamma
    print('gamma files are ' + str([os.path.splitext(os.path.basename(g))[0] for g in path_gamma]))
    # [path_evaluations.remove(g) for g in path_gamma]
    paths['path_evaluations'] = path_evaluations
    print('evaluation files are ' + str([os.path.splitext(os.path.basename(p))[0] for p in path_evaluations]))
    path_roi = folder_selected + '/roi/*.npy'
    path_roi = glob.glob(path_roi)
    paths['path_roi'] = path_roi
    name_rois = [Path(filepath).stem for filepath in paths['path_roi']]
    paths['name_rois'] = name_rois


def plot_settings():
    sns.set_style("white")
    plt.tight_layout()
    sns.set(font_scale=1.05)


def evaluate_gamma(gamma, roi, mask, params, name):
    n = re.search('\d{1,2}\.?\d?', name).group()
    n = float(n)
    if 'gamma' in name:
        n = n * 2
    if 'gammanoise' in name:
        noi = re.search('\d{1,3}\.\d{1,2}', name).group()
        noi = float(noi)
        params['noise'].append(noi)
    else:
        params['noise'].append(0)
    params['name'].append(n)
    roi_bin = roi != 0
    mask_2 = mask & roi_bin
    T2 = 1.5
    gam_tresh = gamma[mask_2]
    valid_gamma = gam_tresh[~np.isnan(gam_tresh)]
    pass_rate = np.round(np.sum(valid_gamma <= 1) / len(valid_gamma) * 100, 2)
    mean_gamma = np.round(np.mean(valid_gamma), 2)  # modified median mean
    T2_gamma = np.sum(valid_gamma > T2) / len(valid_gamma)
    params['valid_gammas'].append(len(valid_gamma))
    params['pass_rate'].append(pass_rate)
    params['mean_gamma'].append(mean_gamma)
    params['T2_gamma'].append(np.round(T2_gamma, 2))


def progress(do_overlay, folder_selected, pat, modus):
    dic_gamma = init()
    output_path = folder_selected + '/analysis_doses'
    os.makedirs(output_path, exist_ok=True)
    print('Selected folder ' + str(folder_selected))
    path_extraction(folder_selected, dicts.paths, folder_selected)

    dose_original = np.load(paths['path_original']) / 100
    doses_modified = [np.load(evaluation) / 100 for evaluation in paths['path_evaluations']]
    gammas = [np.load(g) for g in paths['path_gamma']]
    rois = [np.load(r) for r in paths['path_roi']]

    max_dose = np.max(dose_original)
    dose_cutoff = 0.1 * max_dose
    relevant_slices = (np.max(dose_original, axis=(1, 2)) >= dose_cutoff)
    print('relevant ' + str(relevant_slices.sum()))
    dose_original_cut = dose_original[relevant_slices, :, :]
    print(f'sum {np.sum(dose_original)}')

    [os.makedirs(output_path + '/' + fold, exist_ok=True) for fold in folders.values()]
    if do_gammas:
        external = rois[0]
        rois_2 = []
        names = []
        for roi, name in zip(rois, paths['name_rois']):
            if name == 'External_2':
                external = roi
                print('External')
                print(external.shape)
            else:
                print(name)
                print(roi.shape)
                rois_2.append(roi)
                names.append(name)

        for i, (gamma, dose_modified) in enumerate(zip(gammas, doses_modified)):
            print('gamma shape' + str(gamma.shape))
            gamma_label = f"{gamma_options['distance_mm_threshold']}mm{gamma_options['dose_percent_threshold']}%"
            regex = 'w(\d|\d\d).*.npy'
            name = re.search(regex, paths['path_gamma'][i]).group()[:-4]
            print(f'Evaluating {name} gamma')
            if do_overlay:
                cts = get_cts(pat)
                show_gam = gamma.copy()
                show_gam[show_gam <= 1] = 0
                show_gam[show_gam > 1] = 255
                make_overlay_gamma(pat, cts, rois_2[0], show_gam, modus, name)

            # show_gam[show_gam == np.NAN] = 5
            # data_path_original = r"C:/Users/Carla/OneDrive - tu-dortmund.de/Desktop/Masterarbeit/data/output_data/gauss/png/"
            # if name == 'w4gauss':
            #  Path(f'{data_path_original}{name}').mkdir(parents=True, exist_ok=True)
            #  for i in range(gamma.shape[0]):
            #      if not np.all(show_gam[i, :, :] == 1):
            #        filename_original = f'{data_path_original}{name}/{i}.png'
            #        plt.imsave(filename_original, show_gam[i, :, :])
            # plt.imshow(show_gam[i,:,:,])
            # plt.show()

            # maxi = np.max(dose_modified)
            maxi = prescripted_dose[pat]
            mask = dose_modified > 0.1 * maxi
            evaluate_gamma(gamma, external, mask, dic_gamma, output_path, gamma_label, name)
        df = pd.DataFrame.from_dict(paths.dic_gamma)
        df = df.sort_values(by=['name', 'noise'])
        patient = re.search('zzzCFPatient\d\d', folder_selected).group()
        df.to_csv(folder_selected + '/' + patient + '_' + str(gamma_label) + '_analysis.csv')

        for i, roi in enumerate(rois_2):
            dic_roi = init_roi()
            for j, (gamma, dose_modified) in enumerate(zip(gammas, doses_modified)):
                gamma_label = f"{gamma_options['distance_mm_threshold']}mm{gamma_options['dose_percent_threshold']}%"
                regex = 'w(\d|\d\d).*.npy'
                name = re.search(regex, paths['path_gamma'][j]).group()[:-4]
                print(f'Evaluating {name} gamma roi')
                maxi = prescripted_dose[pat]
                mask = dose_modified > 0.1 * maxi
                evaluate_gamma(gamma, roi, mask, dic_roi, output_path, gamma_label, name, str(i))
            # diff_roi(roi,dic_roi,output_path,dose_original,dose_modified,name,max_dose)
            df = pd.DataFrame.from_dict(dic_roi)
            df = df.sort_values(by=['name', 'noise'])
            patient = re.search('zzzCFPatient\d\d', folder_selected).group()
            df.to_csv(folder_selected + '/' + patient + '_' + str(gamma_label) + '_roi' + names[i] + '_analysis.csv')

