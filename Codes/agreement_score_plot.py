import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

""" 
    Plots figure 5 in the paper, showing the agreement score between the metrics and the observers, for all the experiments.
"""

## Type de scènes blender rendered
type_of_blender_scenes = ["no_bkg_plane", "bkg_plane"]

## Liste des materials rendered
material_list = ["diffuse", "glossy"]

## Liste des méthodes rendered
method_list_indoor = ['gt_indoor', 'everlight', 'weber22', 'gardner19_3', 'stylelight', 'average_image']
method_list_indoor_paths = ['gt_indoor', 'everlight', 'weber22', 'gardner19_3', 'stylelight', 'average_image_texture']
method_list_outdoor = ['gt_outdoor', 'everlight_outdoor', 'jinsong_outdoor', 'average_image_outdoor']
method_list_outdoor_paths = ['gt_outdoor', 'everlight_outdoor', 'jinsong_outdoor', 'average_image_outdoor_texture']

## Liste de métriques
metric_name_list = ["angular_error", "psnr", "rmse", "si_rmse", "ssim", "vif", "pieapp", "flip", "lpips", "delta_E", "hdr_vdp", "brisque", "niqe", "unique", "hyperIQA"]

## Liste des labels des styles des exps
type_of_blender_scenes_label_list = ["Task 1", "Task 2"]
material_label_list = ["Diffuse", "Glossy"]
style_label_list = ["Task 1\nDiffuse", "Task 1\nGlossy", "Task 2\nDiffuse", "Task 2\nGlossy"]
## Liste des labels des méthodes
method_label_list_indoor = ["GT", "EverLight", "Weber", "Gardner", "StyleLight", "Khan"]
method_label_list_outdoor = ["GT", "EverLight", "Zhang", "Khan"]
metric_name_label_list = ["RGB Ang. Err.", "PSNR", "RMSE", "si-RMSE", "SSIM", "VIF", "PieAPP", "FLIP", "LPIPS", r"$\Delta{}E$", "HDR-VDP", "BRISQUE", "NIQE", "UNIQUE", "HyperIQA"]
metric_ranking_list = ["smaller_is_better", "bigger_is_better", "smaller_is_better", "smaller_is_better", "bigger_is_better", "bigger_is_better", "smaller_is_better", "smaller_is_better", "smaller_is_better", "smaller_is_better", "bigger_is_better", "smaller_is_better", "smaller_is_better", "bigger_is_better", "bigger_is_better"]


method_label_ordered_list = ["Weber", "EverLight", "StyleLight", "Gardner", "Khan"]
method_label_ordered_list_outdoor = ["EverLight", "Zhang", "Khan"]
metric_name_label_ordered_list = ["RGB Ang. Err.", "PSNR", "RMSE", "si-RMSE", "SSIM", "VIF", r"$\Delta{}E$", "LPIPS", "PieAPP", "FLIP", "HDR-VDP", "BRISQUE", "NIQE", "UNIQUE", "HyperIQA"]

method_list_paths_ordered = ['gt_indoor', 'weber22', 'everlight', 'stylelight', 'gardner19_3', 'average_image_texture']
method_list_outdoor_paths_ordered = ['gt_outdoor', 'everlight_outdoor', 'jinsong_outdoor', 'average_image_outdoor_texture']

## Path des renders indoor
render_path = Path(f"../Data/Renders/renders_indoor")
## Liste des noms des images
temp_path = render_path / type_of_blender_scenes[0] / "gt_indoor" / material_list[0]
input_im_names_indoor = sorted([f for f in os.listdir(temp_path) if f.endswith('.exr')])

## Path des renders outdoor
render_path = Path(f"../Data/Renders/renders_outdoor")
## Liste des noms des images
temp_path = render_path / type_of_blender_scenes[0] / "gt_outdoor" / material_list[0]
input_im_names_outdoor = sorted([f for f in os.listdir(temp_path) if f.endswith('.exr')])


## Metrics values
def fetch_data_metric_values(domaine, method_list_paths, method_label_list, input_im_names):
    metrics_values_path =Path(f"../Data/Metric_Values/Metrics_Values_{domaine.lower()}/metrics_values_train_{domaine.lower()}.json")

    with open(metrics_values_path) as data_file:    
        harvested_data = json.load(data_file)  

    idx = pd.MultiIndex.from_product([type_of_blender_scenes, material_list, metric_name_list, [str(x) for x in range(len(input_im_names))]],
                                    names=['Type Scene', 'Material', 'Metric', 'Image Name'])
    col = method_list_paths[1:]

    i = 0
    all_data = np.zeros((len(type_of_blender_scenes)*len(material_list)*len(metric_name_list)*len(input_im_names), len(col)))
    for k1 in type_of_blender_scenes:
        v1 = harvested_data[k1]
        for k2 in material_list:
            v2 = v1[k2]
            for k3 in metric_name_list:
                v3 = v2[k3]
                for k4 in [str(x) for x in range(len(input_im_names))]:
                    v4 = v3[k4]
                    all_data[i,:] = list(v4.values())
                    i += 1

    idx = pd.MultiIndex.from_product([type_of_blender_scenes_label_list, material_label_list, metric_name_label_list, input_im_names],
                                    names=['Type Scene', 'Material', 'Metric', 'Image Name'])
    col = method_label_list[1:]
    data_metrics_values = pd.DataFrame(all_data, index=idx, columns=col)

    return data_metrics_values


## Compute the agreement score
harvested_data = {}
for domaine in ["Indoor", "Outdoor"]:
# for domaine in ["Indoor"]:
    for type_of_blender_scene in type_of_blender_scenes_label_list:
        method_list = method_list_indoor if domaine == "Indoor" else method_list_outdoor
        method_list_paths = method_list_indoor_paths if domaine == "Indoor" else method_list_outdoor_paths
        method_label_list = method_label_list_indoor if domaine == "Indoor" else method_label_list_outdoor
        input_im_names = input_im_names_indoor if domaine == "Indoor" else input_im_names_outdoor
        data_metrics_values = fetch_data_metric_values(domaine, method_list_paths, method_label_list, input_im_names)

        for material in material_label_list:
            proportion_preferences_observers_path = Path(f"../Data/Psychophysical_Results/Psychophysical_Results_{domaine.lower()}/{type_of_blender_scenes[type_of_blender_scenes_label_list.index(type_of_blender_scene)]}/{material_list[material_label_list.index(material)]}/proportion_preferences_observers.npy")
            moyenne_arr = np.load(proportion_preferences_observers_path)[0]
            moyenne_arr[moyenne_arr < 0.5] = 0

            for metric in metric_name_label_list:

                ## Master array to hold the preferences for the methods per image (5x5x25)
                rawMatrices = np.ones((len(method_list[1:]), len(method_list[1:]), len(input_im_names))) * np.nan

                for i in range(len(input_im_names)):
                    metric_per_image = data_metrics_values.loc[type_of_blender_scene, material, metric, input_im_names[i]].values

                    ## Compute the preferences de chaque métrique for the methods per image
                    for a, metric_image_a in enumerate(metric_per_image):
                        for b, metric_image_b in enumerate(metric_per_image): 
                            if a != b:
                                if metric_ranking_list[metric_name_label_list.index(metric)] == "smaller_is_better":
                                    if metric_image_a > metric_image_b:
                                        rawMatrices[a, b, i] = 1
                                        rawMatrices[b, a, i] = 0
                                    elif metric_image_a < metric_image_b:
                                        rawMatrices[a, b, i] = 0
                                        rawMatrices[b, a, i] = 1
                                elif metric_ranking_list[metric_name_label_list.index(metric)] == "bigger_is_better":
                                    if metric_image_a > metric_image_b:
                                        rawMatrices[a, b, i] = 0
                                        rawMatrices[b, a, i] = 1
                                    elif metric_image_a < metric_image_b:
                                        rawMatrices[a, b, i] = 1
                                        rawMatrices[b, a, i] = 0


                metrics_results = rawMatrices.transpose(2, 0, 1)
                
                weighted_average = metrics_results*moyenne_arr
                weighted_average_metric = np.nansum(weighted_average) / np.nansum(moyenne_arr)
                harvested_data[(domaine, type_of_blender_scene, material, metric)] = weighted_average_metric

idx = pd.MultiIndex.from_tuples(harvested_data.keys(), names=['Domaine', 'Type Scene', 'Material', 'Metric'])
col = ['Score']

data_metrics = pd.DataFrame(harvested_data.values(), idx, col)


## Plot
colours_domaine = ["#E2975F", "#8CA6BF"]
data = data_metrics
average_observer_score = {("Indoor", "Task 1", "Diffuse"): 0.722416522083219,
                            ("Outdoor", "Task 1", "Diffuse"): 0.792817679558011,
                            ("Indoor", "Task 1", "Glossy"): 0.7584789544799081,
                            ("Outdoor", "Task 1", "Glossy"): 0.8295855379188714,
                            ("Indoor", "Task 2", "Diffuse"): 0.6913246802824965,
                            ("Outdoor", "Task 2", "Diffuse"): 0.6908983451536642,
                            ("Indoor", "Task 2", "Glossy"): 0.7575193174455014,
                            ("Outdoor", "Task 2", "Glossy"): 0.8660904822876655} ## Computed from the raw data
data_observers_correl_both = pd.DataFrame(average_observer_score.values(), index=average_observer_score.keys(), columns=['Score'])
figsize = (11, 5)
fig, axes = plt.subplots(len(type_of_blender_scenes)*len(material_list), 1, figsize=(10*4, 10), sharex=True)
n = 0

for type_of_blender_scene in type_of_blender_scenes_label_list:
    for material in material_label_list:
        data_plot = data.loc[:, type_of_blender_scene, material].unstack(level=0).reindex(index=metric_name_label_ordered_list)
        data_plot.rename(index={"BRISQUE": "BRISQUE*", "NIQE": "NIQE*", "UNIQUE": "UNIQUE*", "HyperIQA": "HyperIQA*"}, inplace=True)

        data_plot['Score'].plot(kind='bar', figsize=figsize, ax=axes[n], color=colours_domaine, legend=False, rot=0, alpha=0.75)

        for i, domaine in enumerate(["Indoor", "Outdoor"]):
            data_plot_obs = data_observers_correl_both.loc[domaine, type_of_blender_scene, material]
            axes[n].bar_label(axes[n].containers[i], label_type='center', fmt='%.3f', rotation=90)
            axes[n].axhline(y=data_plot_obs['Score'], color=colours_domaine[i], linestyle='-', label=f'Expected Observer {domaine}')

        ## Ajouter les lignes du perfect et random observer après les lignes des mean observers pour la legend
        axes[n].axhline(y=[1.0], color='k', alpha=0.5, linestyle='-', label='Perfect')
        axes[n].axhline(y=[0.5], color='k', alpha=0.5, linestyle='-', label='Random')
        axes[n].set_ylabel(f"{type_of_blender_scene}, {material.lower()}")
        axes[n].set_ylim(0, 1.01)
        sns.despine()
        n += 1
axes[0].legend(loc='upper center', borderaxespad=-2, frameon=False, ncol=6)
     
plt.tight_layout()
plt.show()