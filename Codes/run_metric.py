import json
import pickle
import argparse
from pathlib import Path

import numpy as np

from data import preprocess


def load_model(filename):
    with open(str(filename), "rb") as fhdl:
        return pickle.load(fhdl)


def main(model_type, scene_type, material, measures1, measures2):
    assert model_type in ["ours_indoor", "ours_outdoor"], "model_type should be ours_indoor, ours_outdoor"
    assert scene_type in ["no_bkg_plane", "bkg_plane"], "scene_type should be no_bkg_plane or bkg_plane"
    assert material in ["diffuse", "glossy"], "material should be either diffuse or glossy"
    assert isinstance(measures1, list) and len(measures1) >= 15, "measures should be a list of 15 values"
    assert isinstance(measures2, list) and len(measures2) >= 15, "measures should be a list of 15 values"
    print(f"Executing metric for {scene_type} with material {material}")

    model = load_model(list(Path("Models/").glob(f"{model_type}_{scene_type}_{material}_k*.pkl"))[0])

    left = preprocess(np.asarray(measures1)[:10].copy())
    right = preprocess(np.asarray(measures2)[:10].copy())

    x = (left - right).reshape((1, -1))
    prediction = np.clip(model.predict(x).item(), 0, 1)

    print("""Score interpretation:
        0 = left estimated to be always preferred by humans
        0.5 = humans are confused and would randomly prefer either image
        1 = right estimated to be always preferred by humans\n""")
    print(f"Perceptual Metric score: {prediction:.03f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python run_metric.py",
                                     description="Compares two images based on the measures computed from their sphere+plane renders",
                                     epilog="Expected measure order is: [rmse, si_rmse, angular_error, psnr, ssim, vif, pieapp, flip, lpips, delta_E]")
    parser.add_argument('--model_type', default="ours_indoor", choices=["ours_indoor", "ours_outdoor"])
    parser.add_argument('--scene_type', default="bkg_plane", choices=["no_bkg_plane", "bkg_plane"])
    parser.add_argument('--material', default="diffuse", choices=["diffuse", "glossy"])
    parser.add_argument('measures1', type=str, help="Measures from the first sphere+plane render in json format")
    parser.add_argument('measures2', type=str, help="Measures from the second sphere+plane render in json format")
    args = parser.parse_args()

    with open(args.measures1, "r") as fhdl:
        measures1 = json.load(fhdl)

    with open(args.measures2, "r") as fhdl:
        measures2 = json.load(fhdl)

    main(args.model_type, args.scene_type, args.material, measures1, measures2)
