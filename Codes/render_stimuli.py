import os
from tqdm import tqdm
import json
from pathlib import Path

import numpy as np
import cv2
from skimage import io
from skimage import img_as_float, img_as_ubyte

from ezexr import imread


def run():
    """ Render the stimuli with a new lighting estimation method.  
    ``dir_path`` needs to be changed to be the path to the current directory.
    ``domaine`` is set to either "indoor" or "outdoor" depending on the domaine of the lighting estimation method and the crops used to predict the envmaps.
    ``model_name`` is the name of the new model used to predict the envmaps (will be used as the name of the dir).
    ``predicted_envmaps_path`` is the path to the dir of the predicted envmaps with the new lighting estiamtion method.  The predicted envmaps must have the same name as the crops (without the "_crop" for the indoor crops).
    ``path_app_blender`` is the path of the blender app (requires version 3.4.1).
    """
    ## Path to change
    dir_path = Path('/Users/justinegiroux/Desktop/Lightsome_Public/')
    domaine = "indoor" #"outdoor"
    model_name = "qwerty"
    predicted_envmaps_path = Path('/Users/justinegiroux/Documents/Universite/Maitrise_GEL/Metric_Project/Data/Envmaps/gt_indoor_envmaps')
    path_app_blender = "/Applications/Blender.app/Contents/MacOS/Blender"


    ## Path to the python blender script
    path_blender_script = dir_path / "Utils/script_render_single_spheres_with_envmap_materials.py"
    
    ## Path of the directory to save the renders
    output_render_path = dir_path / Path(f"Data/Renders/renders_{domaine}")

    ## Path du directory des crops pour les bkgs des renders
    input_crops_bkg_dir = dir_path / Path(f"Data/Crops/crops_{domaine}")

    ## Name of the images to render
    input_im_names = sorted([f for f in os.listdir(input_crops_bkg_dir) if f.endswith('.png')])
    if domaine == "indoor":
        input_im_names = [s.replace('_crop.png', '.exr') for s in input_im_names] ## Remove the "_crop" from the names of the indoor crops and get the envmap names (.exr)
    elif domaine == "outdoor":
        input_im_names = [s.replace('.png', '.exr') for s in input_im_names]
    input_im_names = input_im_names[0:5]

    ## Liste des materials à render
    material_list = ["diffuse", "glossy"]

    ## Type de scènes blender à render
    type_of_blender_scenes = ["no_bkg_plane", "bkg_plane"]



    def tonemap(img, gamma=2.4):
        """Apply gamma, then clip between 0 and 1, finally convert to uint8 [0,255]"""
        return (np.clip(np.power(img,1/gamma), 0.0, 1.0)*255).astype('uint8')

    def reexpose_hdr_mask(hdrim, percentile=90, max_mapping=0.8, alpha=None):
        """
        Code de create_weber_test_set.py, mais utilise nanpercentile pour prendre en compte les nans du background transparent (sinon l'exposition est wrong)
        :param img:         HDR image
        :param percentile:
        :param max_mapping:
        :return:
        """
        r_percentile = np.nanpercentile(hdrim, percentile)
        if alpha==None:
            alpha = max_mapping / (r_percentile + 1e-10)
        return alpha * hdrim, alpha



    ## Render the stimuli
    for type_of_blender_scene in type_of_blender_scenes:
        
        ## Get the path to the blender file associated to the type of blender scene
        if type_of_blender_scene == "no_bkg_plane":
            blender_file_path = Path("Utils/blender_scenes/psychophysical_experiment_single_sphere_no_bkg_plane.blend")
        elif type_of_blender_scene == "bkg_plane":
            blender_file_path = Path("Utils/blender_scenes/psychophysical_experiment_single_sphere_bkg_plane.blend")

        for material in material_list:
            
            n = 0
            for input_im_name in tqdm(input_im_names[:5]):
                output_render_path_experiment = dir_path / output_render_path / type_of_blender_scene / model_name / material
                ## Create the folder to save the renders of the new method
                output_render_path_experiment.mkdir(parents=True, exist_ok=True)

                ## Render the image
                cmd = f"{path_app_blender} -b -P {path_blender_script} -- " + str(output_render_path_experiment) + \
                        " " + str(predicted_envmaps_path) + " " + input_im_name + " " + str(blender_file_path) + " " + material
                os.system(cmd + " > /dev/null 2>&1")

                ## Read HDR render 
                print(cmd)
                print(os.path.exists(str(output_render_path_experiment / input_im_name)))
                render = imread(str(output_render_path_experiment / input_im_name))
                render_mask = render[:,:,-1].copy()

                ## Mask le bkg transparent des renders pour que l'exposition soit correcte et uniforme entre les différentes scènes
                mask_bkg = render[:,:,-1] == 0
                mask_bkg = np.dstack((mask_bkg, mask_bkg, mask_bkg))
                render_mask_binary = render[:,:,:-1].copy()
                render_mask_binary[mask_bkg] = np.nan

                ## Reexpose render
                render, _ = reexpose_hdr_mask(render_mask_binary)

                ## Tonemap render
                render = tonemap(render)  ## render est en UINT
                

                ## Write LDR render
                render = img_as_ubyte(render) ## Convert en UINT pour l'écriture
                cv2.imwrite(os.path.join(output_render_path_experiment, input_im_name[:-4] + ".png"),  cv2.cvtColor(render, cv2.COLOR_RGB2BGR))

                ## Composite the render with the background
                if type_of_blender_scene == "bkg_plane":
                    ## Read the background crop
                    if domaine == "indoor":
                        bkg_im = io.imread(os.path.join(input_crops_bkg_dir, input_im_name[:-4] + "_crop.png"))
                    elif domaine == "outdoor":
                        bkg_im = io.imread(os.path.join(input_crops_bkg_dir, input_im_name[:-4] + ".png"))

                    bkg_im = cv2.resize(bkg_im, (render.shape[1], render.shape[0]))
                    bkg_im = img_as_float(bkg_im)
                    render = img_as_float(render)

                    M = np.dstack([render_mask] * 3)
                    R = render

                    render_bkg = M * R + (1-M) * bkg_im
                    render_bkg = np.clip(255.*render_bkg, 0, 255).astype(np.uint8)

                    cv2.imwrite(os.path.join(output_render_path_experiment, input_im_name[:-4] + ".png"), cv2.cvtColor(render_bkg, cv2.COLOR_RGB2BGR))

                n += 1


if __name__ == "__main__":
    run()
