# Data

## Crops
Required to render the stimuli with ``render_stimuli.py``.
The crops extracted from the ULaval dataset used as input to the lighting estimation methods and as background to the stimuli.

## Renders
Produced by ``render_stimuli.py``.
The stimuli used in all the psychophysical experiments, for the different methods.

## Metric_Values
Used to train the metrics (``metrics_values_train_indoor.json`` and ``metrics_values_train_outdoor.json``).  ``metrics_values_indoor.json`` and ``metrics_values_outdoor.json`` are the values computed by ``compute_metric_values.py``, using the new lighting estimation model.
The scores of the 15 metrics computed on all the stimuli used in the experiments.  The scores are stored in a ``.json`` containing all the experiments.  The hirarchy is:
1. task: ["no_bkg_plane", "bkg_plane"] (which corresponds to task 1 and task 2)
2. material: ["diffuse", "glossy"]
3. metric name: ["rmse", "si_rmse", "angular_error", "psnr", "ssim", "vif", "pieapp", "flip", "lpips", "delta_E", "hdr_vdp", "brisque", "niqe", "unique", "hyperIQA"]
4. image: str with a number from 0 to 25 corresponding to the crops in alphabetical order
5. method: ['gt_indoor', 'everlight', 'weber22', 'gardner19_3', 'stylelight', 'average_image_texture'] (indoor) or ['gt_outdoor', 'everlight_outdoor', 'jinsong_outdoor', 'average_image_outdoor_texture'] (outdoor)

## Psycophysical_Results
### ``psychological_results.mat``
Raw choices of all the observers for the specific experiment, stored in a 4D matrix ``.mat``.  The hirarcy is:
1. observer: between 30-31 for the indoor experiments and 12 for the outdoor expriments
2. image: in alphabetical order of the crops' names
3. choices: the 2D matrix corresponds to the choices of that observer for that image.  The matrix's rows and columns correspond to ['everlight', 'weber22', 'gardner19_3', 'stylelight', 'average_image_texture'] or ['everlight_outdoor', 'jinsong_outdoor', 'average_image_outdoor_texture'], resulting in a 5x5 (indoor) or 3x3 (outdoor) matrix.  Each stimulus pair shown receives a 0 for the method not prefered and a 1 for the method prefered by that observer.  A 1 in the method's column indicates that this method was picked.  The diag of the matrix is nans, as observers are not asked to compare the method with itself (identical stimuli).

### ``results.mat``
Required for the figure generated in ``lighting_estimation_methods_rating_with_perceptual_data.py``.
The Thurstone Case V Law of Comparative Judgement (z-scores) computed from the choices of all the observers for all the images.  The scores are stored in a dict (``.mat``), in a vector for each method, in the order of ['gt_indoor', 'everlight', 'weber22', 'gardner19_3', 'stylelight', 'average_image_texture'] (indoor) or ['gt_outdoor', 'everlight_outdoor', 'jinsong_outdoor', 'average_image_outdoor_texture'] (outdoor).  The ``'resultsOverall'`` vector corresponds to the score and ``'error_bars'`` to the error associated to the score.

### ``results_each_image.mat``
The Thurstone Case V Law of Comparative Judgement (z-scores) for each method (same order as ``results.mat``) and each image (in alphabetical order of the crops' names).

### ``proportion_preferences_observers.npy``
Produced by ``psychological_results.mat``.  Used to train the metrics.
Proportion of observers that made the same choice, obtained by computing the mean on the observers (axis=0) of ``psychological_results.mat``.  For example, the value indicates that 80% of the observers picked method A (column) over method B (row), which would mean that by symmetry, 20% picked method B (column) over method A (row).
The proportions are stored in a 3D matrix (``.mat``) for each experiment.  The hirarchy is:
1. image: in alphabetical order of the crops' names
2. proportions of the choices: the 2D matrix corresponds to the proportion of the choices made by all the observers for that image.  The matrix's rows and columns correspond to ['everlight', 'weber22', 'gardner19_3', 'stylelight', 'average_image_texture'] or ['everlight_outdoor', 'jinsong_outdoor', 'average_image_outdoor_texture'], resulting in a 5x5 (indoor) or 3x3 (outdoor) matrix.  The values are continuous between [0, 1].


# Codes
To be able to run the metric, you must generate the renders (``render_stimuli.py``), compute the metric values (``compute_metric_values.py``), and then run our metric on the metric values (``run_metric.py``).

## ``render_stimuli.py``
Requires the crops to composite in the background.
Required for ``compute_metric_values.py``.
Render the stimuli with a new lighting estimation method.  
``dir_path`` needs to be changed to be the path to the current directory.
``domaine`` is set to either "indoor" or "outdoor" depending on the domaine of the lighting estimation method and the crops used to predict the envmaps.
``model_name`` is the name of the new model used to predict the envmaps (will be used as the name of the dir).
``predicted_envmaps_path`` is the path to the dir of the predicted envmaps with the new lighting estiamtion method.  The predicted envmaps must have the same name as the crops (without the "_crop" for the indoor crops).
``path_app_blender`` is the path of the blender app (requires version 3.4.1).
To install ezexr:
``conda install -c conda-forge openexr-python openexr``
``pip install --upgrade skylibs``
See https://github.com/soravux/skylibs for more information.

## ``compute_metric_values.py``
Requires the renders generated by ``render_stimuli.py`` and the values computed by ``compute_metric_values_matlab.py`` (for HDR-VDP3 and NIQE metric).
Required for ``run_metric.py``.
Compute the values of 15 metrics for a render.  The Full-Reference metrics use the ground truth render associated to the same crop used by the render generated with the new predicted envmap (they should have the same file name).  The No-Reference metrics only use the render generated by that method.
The list of metrics with their reference and the code required to run them.
RMSE: implemented in ``compute_metric_values.py``
si-RMSE: implemented in ``compute_metric_values.py``
RGB angular error: implemented in ``compute_metric_values.py``
PSNR: implemented in ``compute_metric_values.py``
SSIM: https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html
VIF: https://github.com/abhinaukumar/vif
PieAPP: https://github.com/prashnani/PerceptualImageError
FLIP: https://github.com/NVlabs/flip
LPIPS: https://github.com/richzhang/PerceptualSimilarity
Delta E: implemented in ``compute_metric_values.py``
BRISQUE: https://github.com/krshrimali/No-Reference-Image-Quality-Assessment-using-BRISQUE-Model
UNIQUE: https://github.com/zwx8981/UNIQUE
HyperIQA: https://github.com/SSL92/hyperIQA
HDR-VDP3: see ``compute_metric_values_matlab.py``
NIQE: ``compute_metric_values_matlab.py``


Outputs the file ``metrics_values_indoor.json`` or ``metrics_values_outdoor.json`` in ``Data/Metric_Values/Metric_Values_*door/``.

## ``compute_metric_values_matlab.py``
The matlab code to run the HDR-VDP3 metric and NIQE.
``baseFolder_dir`` is the path to the renders generated by ``render_stimuli.py``.
``domaine`` is set to either "indoor" or "outdoor" depending on the domaine of the lighting estimation method and the crops used to predict the envmaps.
The values are then used by ``compute_metric_values.py`` to generate a ``.json``file with all the metric values.
HDR-VDP3: https://github.com/ug-kim/hdr-vdp
NIQE: https://www.mathworks.com/help/images/ref/niqe.html



## ``run_metric.py``
Requires the renders generated by ``render_stimuli.py``, to obtain the metric values computed by ``compute_metric_values.py``.
Compare two renders in order to determine the one prefered by our metric, trained on the perceptual data.
``model_type`` is the domaine of the lighting estimation method and the crops used to predict the envmaps, either "ours_indoor" or "ours_outdoor"
``scene_type`` is the task of the render given as input, either "no_bkg_plane" (task 1) or "bkg_plane" (task 2).
``material`` is the material of the render given as input, either "diffuse" or "glossy".
``mesures1`` is the ``.json`` file of the metric values computed for a render (with reference to its associated ground truth render)
``mesures2`` is the ``.json`` file of the metric values computed for the render to be compared to the first render, computed in the same way as ``measures1``.
Outputs a score, where a value in [0, 0.5[ indicates that the render used for ``mesures1`` is preferred by humans and ]0.5, 1] indicates that the render used fot ``mesures2`` is prefered by humans.  A score near 0.5 indicates that humans are confused and would randomly prefer either image.




## ``lighting_estimation_methods_rating_with_perceptual_data.py``
Requires ``results.mat`` to generate figure 4. of the paper.
Plots figure 4 in the paper, showing the prefered lighting estimation methods by the observers, according to the Thurstone Case V Law of Comparative Judgement (z-score), for all the experiments.
``dir_path`` needs to be changed to be the path to the current directory.


