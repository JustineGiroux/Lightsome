import json
import os
from pathlib import Path
import subprocess
from math import log10

import colour
import lpips as lpips_compute
import numpy as np
import pandas as pd
import torch
from ezexr import imread
from metrics_code.vif.vif_utils import vif as vif_compute
from scipy.io import loadmat
from skimage import color, img_as_float, io, metrics
from skimage.transform import resize
# import cv2 as cv
from brisque import BRISQUE
from metrics_code.UNIQUE.compute_score import unique as unique_compute

## Metrics
def wrmse(gt, est, mask=None):
    if mask is None:
        gt = gt.flatten()
        est = est.flatten()
    else:
        gt = gt[mask].flatten()
        est = est[mask].flatten()
    error = np.sqrt(np.mean(np.power(gt - est, 2)))

    return error

def si_wrmse(gt, est, mask=None):
    if mask is None:
        gt_c = gt.flatten()
        est_c = est.flatten()
    else:
        gt_c = gt[mask].flatten()
        est_c = est[mask].flatten()
    alpha = (np.dot(np.transpose(gt_c), est_c)) / (np.dot(np.transpose(est_c), est_c))
    error = wrmse(gt, est * alpha, mask)

    return error

def angular_error(gt_render, pred_render, mask=None):
    # The error need to be computed with the normalized rgb image.
    # Normalized RGB is r = R / (R+G+B), g = G / (R+G+B), b = B / (R+G+B)
    # The angular distance is the distance between pixel 1 and pixel 2.
    # It's computed with cos^-1(p1·p2 / ||p1||*||p2||)
    gt_norm = np.empty((gt_render.shape))
    pred_norm = np.empty(pred_render.shape)

    for i in range(3):
        gt_norm[:,:,i] = gt_render[:,:,i] / np.sum(gt_render, axis=2, keepdims=True)[:,:,0]
        pred_norm[:,:,i] = pred_render[:,:,i] / (np.sum(pred_render, axis=2, keepdims=True)[:,:,0] + 1e-8)

    angular_error_arr = np.arccos( np.sum(gt_norm*pred_norm, axis=2, keepdims=True)[:,:,0] / 
        ((np.sqrt(np.sum(gt_norm*gt_norm, axis=2, keepdims=True)[:,:,0])*np.sqrt(np.sum(pred_norm*pred_norm, axis=2, keepdims=True)[:,:,0]))) )

    if mask is not None:
        angular_error_arr = angular_error_arr[mask[:,:,0]]
    else:
        angular_error_arr = angular_error_arr.flatten()
    angular_error_arr = angular_error_arr[~np.isnan(angular_error_arr)]
    mean = np.mean(angular_error_arr)
    # convert to degree
    mean = mean * 180 / np.pi
    return mean

def psnr(original, compressed):
    ## Les images sont en float64 [0, 1] --> convert [0, 255]
    original = np.clip(255.*original, 0, 255)
    compressed = np.clip(255.*compressed, 0, 255)
    
    mse = wrmse(original, compressed, None)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255
    psnr_value = 20 * log10(max_pixel / mse)
    return psnr_value

def ssim(original, compressed):
    ssim_value = metrics.structural_similarity(original, compressed, data_range=(compressed.max()-compressed.min()), channel_axis=2)
    return ssim_value

def vif(original, compressed):
    vif_R = vif_compute(original[:,:,0], compressed[:,:,0])
    vif_G = vif_compute(original[:,:,1], compressed[:,:,1])
    vif_B = vif_compute(original[:,:,2], compressed[:,:,2])
    vif_value = np.mean([vif_R, vif_G, vif_B])
    return vif_value

def pieapp(original_path, compressed_path):
    cmd = f"wine64 PieAPPv0.1.exe --ref_path /Users/justinegiroux/Documents/Universite/Maitrise_GEL/Metric_Project/{original_path} --A_path /Users/justinegiroux/Documents/Universite/Maitrise_GEL/Metric_Project/{compressed_path} --sampling_mode sparse"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd="/Users/justinegiroux/Documents/Universite/Maitrise_GEL/Metric_Project/evaluation_pipeline/scripts/quantitative/metrics_code/PerceptualImageError/PieAPPv0.1_win64_exe/")
    pieapp_value, err = process.communicate()
    process.wait()
    return float(pieapp_value.split(b"\r\n")[-2])

def flip(original_path, compressed_path):
    cmd = f"python flip.py --reference /Users/justinegiroux/Documents/Universite/Maitrise_GEL/Metric_Project/{original_path} --test /Users/justinegiroux/Documents/Universite/Maitrise_GEL/Metric_Project/{compressed_path}"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd="/Users/justinegiroux/Documents/Universite/Maitrise_GEL/Metric_Project/evaluation_pipeline/scripts/quantitative/metrics_code/flip/python/")
    flip_value, err = process.communicate()
    process.wait()
    return float(flip_value.split(b"\tMean: ")[-1].split(b"\n")[0])

def lpips(original, compressed): 
    original_normalised = (original - original.max()/2)/ (original.max()/2)
    compressed_normalised = (compressed - compressed.max()/2) / (compressed.max()/2)
    original_normalised = resize(original_normalised, (3, 256, 256))
    compressed_normalised = resize(compressed_normalised, (3, 256, 256))
    original_normalised = original_normalised.astype(np.float32)
    compressed_normalised = compressed_normalised.astype(np.float32)
    original_normalised = torch.from_numpy(original_normalised)
    compressed_normalised = torch.from_numpy(compressed_normalised)
    loss_fn = lpips_compute.LPIPS(net='alex')
    lpips_value = loss_fn.forward(original_normalised, compressed_normalised)
    return lpips_value.item()


def delta_E(original, compressed):
    original = color.rgb2lab(original)
    compressed = color.rgb2lab(compressed)
    delta_E = colour.delta_E(original, compressed)
    return np.mean(delta_E)

def hyperIQA(_, compressed_path):
    cmd = f"python hyperiqa.py /Users/justinegiroux/Documents/Universite/Maitrise_GEL/Metric_Project/{compressed_path}"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd="/Users/justinegiroux/Documents/Universite/Maitrise_GEL/Metric_Project/evaluation_pipeline/scripts/quantitative/metrics_code/HyperIQA/")
    hyperIQA_score, err = process.communicate()
    process.wait()
    return float(hyperIQA_score.split(b"Predicted quality score: ")[-1].split(b"\n")[0])

def unique(_, compressed_path):
    return unique_compute(compressed_path)

def brisque(_, compressed):
    obj = BRISQUE(url=False)
    return obj.score(compressed)

## Dummy function for metrics that require running on matlab
def niqe(_):
    return None

## Dummy function for metrics that require running on matlab
def hdr_vdp(_):
    return None


""" Compute the values of the metrics for the renders generated by ``render_stimuli.py``.  
``dir_path`` needs to be changed to be the path to the current directory.
``domaine`` is set to either "indoor" or "outdoor" depending on the domaine of the lighting estimation method and the crops used to predict the envmaps.
``model_name`` is the name of the new model used to predict the envmaps (will be used as the name of the dir).
"""

## Path to change
dir_path = Path('/Users/justinegiroux/Documents/Universite/Maitrise_GEL/Metric_Project/')
domaine = "indoor" #"outdoor"
model_name = ["qwerty"]

## Path des renders
render_path = dir_path / Path(f"Data/Renders/renders_{domaine}")
output_root_path = dir_path / Path(f"Data/Metrics_Results/Metrics_Values/Metrics_Values_{domaine}")

## Type de scènes blender rendered
type_of_blender_scenes = ["no_bkg_plane", "bkg_plane"]

## Liste des materials rendered
material_list = ["diffuse", "glossy"]

## Liste des metrics
metric_list = [wrmse, si_wrmse, angular_error, psnr, ssim, vif, pieapp, flip, lpips, delta_E, hdr_vdp, brisque, niqe, unique, hyperIQA]
metric_name_list = ["rmse", "si_rmse", "angular_error", "psnr", "ssim", "vif", "pieapp", "flip", "lpips", "delta_E", "hdr_vdp", "brisque", "niqe", "unique", "hyperIQA"]
metric_ranking_list = ["smaller_is_better", "smaller_is_better", "smaller_is_better", "bigger_is_better", "bigger_is_better", "bigger_is_better", "smaller_is_better", "smaller_is_better", "smaller_is_better", "smaller_is_better", "bigger_is_better", "smaller_is_better", "smaller_is_better", "bigger_is_better", "bigger_is_better"]
metric_requires_path = [False, False, False, False, False, False, True, True, False, False, False, False, False, True, True]
extension_list = [".png", ".png", ".png", ".png", ".png", ".png", ".png", ".png", ".png", ".png", ".png", ".png", ".png", ".png", ".png"]


## Liste des noms des images
temp_path = render_path / type_of_blender_scenes[0] / model_name / material_list[0]
input_im_names = sorted([f for f in os.listdir(temp_path) if f.endswith('.exr')])

### Create dict to save the values of the metrics
dict_metrics_values = {}

for type_of_blender_scene in type_of_blender_scenes:
    dict_metrics_values[f'{type_of_blender_scene}'] = {} 
    for material in material_list:
        dict_metrics_values[f'{type_of_blender_scene}'][f'{material}'] = {} 
        for metric in metric_list:
            dict_metrics_values[f'{type_of_blender_scene}'][f'{material}'][f'{metric_name_list[metric_list.index(metric)]}'] = {} 
                        
            ## Master array to hold the preferences for the methods per image (5x5x25)
            rawMatrices = np.ones((len(model_name), len(model_name), len(input_im_names))) * np.nan
            
            ## Case for the metrics that require running on matlab (HDR-VDP and NIQE)
            if (metric == hdr_vdp) or (metric == niqe):
                matlab_values_path = dir_path / Path(f"Data/Metrics_Results/Metrics_Values/Metrics_Values_{domaine}/{type_of_blender_scene}/{material}/{metric_name_list[metric_list.index(metric)]}/{metric_name_list[metric_list.index(metric)]}_values.mat")
                matlab_values = loadmat(matlab_values_path)['harvested_data']
                matlab_values = pd.DataFrame(matlab_values)
                
                for i in range(len(input_im_names)):
                    metric_per_image = matlab_values.loc[i].to_list()
                    ### Save the values
                    dict_metrics_values[f'{type_of_blender_scene}'][f'{material}'][f'{metric_name_list[metric_list.index(metric)]}'][f'{i}'] = dict(zip(model_name, metric_per_image))
            
            ## For the metrics ran with python
            else:
                for i in range(len(input_im_names)):
                    ## Fetch le path de l'image de GT pour la bonne extension
                    image_gt_path = render_path / type_of_blender_scene / f"gt_{domaine}" / material / input_im_names[i][:-4] + extension_list[metric_list.index(metric)] 
                    ## Fetch les images de chaque méthode pour la bonne extension
                    if metric_requires_path[metric_list.index(metric)]: ## Si la fct de la métrique prend en input des paths
                        metric_per_image = [metric(image_gt_path, os.path.join(render_path, type_of_blender_scene, model_name[j], material, input_im_names[i][:-4] + extension_list[metric_list.index(metric)])) for j in range(len(model_name))]

                    else: ## Si la fct de la métrique prend en input des ims
                        if extension_list[metric_list.index(metric)] == ".exr": ## Si la métrique prend en input des exr
                            image_gt = imread(image_gt_path)[:,:,:3].astype(np.float64)
                            metric_per_image = [metric(image_gt, imread(os.path.join(render_path, type_of_blender_scene, model_name[j], material, input_im_names[i]))[:,:,:3].astype(np.float64)) for j in range(len(model_name))]
                        elif extension_list[metric_list.index(metric)] == ".png": ## Si la métrique prend en input des png
                            image_gt = img_as_float(io.imread(image_gt_path))[:,:,:3].astype(np.float64)
                            metric_per_image = [metric(image_gt, img_as_float(io.imread(os.path.join(render_path, type_of_blender_scene, model_name[j], material, input_im_names[i][:-4] + ".png"))[:,:,:3]).astype(np.float64)) for j in range(len(model_name))]

                    ### Add the values to the master array
                    dict_metrics_values[f'{type_of_blender_scene}'][f'{material}'][f'{metric_name_list[metric_list.index(metric)]}'][f'{i}'] = dict(zip(model_name, metric_per_image))

### Save the values to a json file
with open(os.path.join(output_root_path, f"metrics_values_{domaine}.json"), "w") as outfile:
    json.dump(dict_metrics_values, outfile)