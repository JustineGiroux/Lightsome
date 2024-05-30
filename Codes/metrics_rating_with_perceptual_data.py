from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

""" 
    Plots figure 4 in the paper, showing the prefered lighting estimation methods by the observers, according to the Thurstone Case V Law of Comparative Judgement (z-score), for all the experiments.
"""

## Scores des observateurs
observers_results_dir = Path('../Data/Psychophysical_Results/Psychophysical_Results_indoor/')
observers_results_outdoor_dir = Path('../Data/Psychophysical_Results/Psychophysical_Results_outdoor/')

## Type de scènes blender rendered
type_of_blender_scenes = ["no_bkg_plane", "bkg_plane"]

## Liste des materials rendered
material_list = ["diffuse", "glossy"]

## Liste des méthodes rendered
method_list = ['gt_indoor', 'everlight', 'weber22', 'gardner19_3', 'stylelight', 'average_image']
method_list_outdoor = ['gt_outdoor', 'everlight_outdoor', 'jinsong_outdoor', 'average_image_outdoor']

## For the plot
## Liste des labels des styles des exps
type_of_blender_scenes_label_list = ["Task 1", "Task 2"]
material_label_list = ["Diffuse", "Glossy"]

## Liste des labels des méthodes
method_label_list = ["GT", "EverLight", "Weber", "Gardner", "StyleLight", "Khan"]
method_label_list_outdoor = ["GT", "EverLight", "Zhang", "Khan"]

method_label_ordered_list = ["Weber", "EverLight", "StyleLight", "Gardner", "Khan"]
method_label_ordered_list_outdoor = ["EverLight", "Zhang", "Khan"]

## Colours
colours_methods = ["#8CA6BF", "#E2975F", "#8C8D62", "#D8A728", "#C97676"]
colours_methods_outdoor = ["#E2975F", "#AFD4AF", "#C97676"]


## Load the data
## Data des observateurs indoor
harvested_data = []
for type_of_blender_scene in type_of_blender_scenes:
    image_path_scene = observers_results_dir / type_of_blender_scene
    for material in material_list:
        image_path_material = image_path_scene / material
        
        results_path = image_path_material / "results.mat"
        matlab_results = loadmat(results_path)
        results = matlab_results['resultsOverall']
        error_bars = matlab_results['error_bars']
        results_all = np.vstack((results, error_bars)).T
        harvested_data.append(results_all)

harvested_data = np.array(harvested_data)
harvested_data = harvested_data.reshape((len(type_of_blender_scenes)*len(material_list)*len(method_list[1:]), 2))

idx = pd.MultiIndex.from_product([type_of_blender_scenes_label_list, material_label_list, method_label_list[1:]],
                                 names=['Type Scene', 'Material', 'Method'])
col = ['Score', 'Error']

data_observers_indoor = pd.DataFrame(harvested_data, idx, col)

## Data des observateurs outdoor
harvested_data = []
for type_of_blender_scene in type_of_blender_scenes:
    image_path_scene = observers_results_outdoor_dir / type_of_blender_scene
    for material in material_list:
        image_path_material = image_path_scene / material
        
        results_path = image_path_material / "results.mat"
        matlab_results = loadmat(results_path)
        results = matlab_results['resultsOverall']
        error_bars = matlab_results['error_bars']
        results_all = np.vstack((results, error_bars)).T
        harvested_data.append(results_all)

harvested_data = np.array(harvested_data)
harvested_data = harvested_data.reshape((len(type_of_blender_scenes)*len(material_list)*len(method_list_outdoor[1:]), 2))

idx = pd.MultiIndex.from_product([type_of_blender_scenes_label_list, material_label_list, method_label_list_outdoor[1:]],
                                 names=['Type Scene', 'Material', 'Method'])
col = ['Score', 'Error']

data_observers_outdoor = pd.DataFrame(harvested_data, idx, col)

## Plot
## Barplot des scores des observateurs indoor et outdoor
data_indoor = data_observers_indoor
data_outdoor = data_observers_outdoor

data_list = [data_indoor, data_outdoor]
data_in_and_out = pd.concat([data_indoor, data_outdoor], axis=0, keys=['Indoor', 'Outdoor'], names=["Domaine"])
colours_methods_in_and_out = [colours_methods, colours_methods_outdoor]
figsize = (10, 4.5)

fig = plt.figure(figsize=figsize, constrained_layout=True)
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

for type_of_blender_scene in type_of_blender_scenes_label_list:
    middle = gridspec.GridSpecFromSubplotSpec(1, 2,
                        subplot_spec=outer[type_of_blender_scenes_label_list.index(type_of_blender_scene)], wspace=0.2, hspace=0.2, width_ratios=[5/8, 3/8])
    for i in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(len(material_list), 1,
                        subplot_spec=middle[i], wspace=0.1, hspace=0.1)
        data = data_list[i%2]
        n = 0
        for material in material_label_list:
            ax = plt.Subplot(fig, inner[n])

            data_plot = data.loc[type_of_blender_scene, material].unstack(level=0)
            if i%2 == 0:
                data_plot = data_plot.reindex(method_label_ordered_list, level=1)
            if i%2 == 1:
                data_plot = data_plot.reindex(method_label_ordered_list_outdoor, level=1)
            bars = ax.bar(data_plot['Score'].index, data_plot['Score'], color=colours_methods_in_and_out[i%2], width=0.8, yerr=data_plot['Error'])

            if n == 1:
                ax.set_xticklabels(data_plot['Score'].index, rotation=30)
            else:
                ax.set_xticklabels([])

            ax.bar_label(bars, label_type='edge', fmt='%.3f')
            
            if i%2 == 0: ## Si est en train de faire le subplot de gauche (pour pas repeat les labels)
                ax.set_ylabel(f"{type_of_blender_scene}, {material.lower()}")
            ax.axhline(y=0, color='k', alpha=0.5, linestyle='-')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.tick_params(bottom=False)
            if n == 1:
                if i%2 == 0:
                    ax.set_xlabel("Indoor methods")
                if i%2 == 1:
                    ax.set_xlabel("Outdoor methods")
            ax.set_ylim([data_in_and_out.loc[:, type_of_blender_scene, material]["Score"].min()-0.4, data_in_and_out.loc[:, type_of_blender_scene, material]["Score"].max()+0.4])
            sns.despine(bottom=True, ax=ax)
            fig.add_subplot(ax)
            n += 1
fig.tight_layout()
fig.show()