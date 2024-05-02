import argparse
import numpy as np
import os
import yaml
from argparse import Namespace
from faster_bootstrap import get_dataset, seg_results, gen_results, statistics, save_as_csv

def get_args():
    parser = argparse.ArgumentParser("argument for bootstrap")

    parser.add_argument("--model", "-m", type=str, default="unet_plus_plus",
                        help="unet, unet_plus_plus, unet_plus_plus_3d, unet_att, genunet")
    parser.add_argument("--model_name", type=str, default="unet_plus_plus_2024-04-02/")
    parser.add_argument("--version", "-v",  type=int, default=1)
    args = parser.parse_args()
    return args

def volume_bootstrap(config, model_path, gen=False):
    test_IDs = np.array(["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"])
    res_dict = {}

    for ID in test_IDs:
        if gen:
            res = gen_results(config, model_path, [ID])
        else:
            res = seg_results(config, model_path, [ID])
        res_dict[ID] = res

    dasboot = {"Dice": np.zeros((len(test_IDs), 3)),
               "IoU": np.zeros((len(test_IDs), 3)),
               "Hausdorff": np.zeros((len(test_IDs), 3))}

    for i in range(len(test_IDs)):
        leave_out = test_IDs[i]
        sample = test_IDs[test_IDs != leave_out]

        dice = []
        iou = []
        hausdorff = []
        for ID in sample:
            res = res_dict[ID]
            dice.append(res["Dice"])
            iou.append(res["IoU"])
            hausdorff.append(res["Hausdorff"])

        dice = np.concatenate(dice, axis=0)
        iou = np.concatenate(iou, axis=0)
        hausdorff = np.concatenate(hausdorff, axis=0)

        dasboot["Dice"][i] = np.nanmean(dice, axis=0)
        dasboot["IoU"][i] = np.nanmean(iou, axis=0)
        dasboot["Hausdorff"][i] = np.nanmean(hausdorff, axis=0)

    return dasboot

def load_results(model_path):
    with open(os.path.join(model_path, "volume_bootstrap.yaml"), "r") as f:
        res = yaml.unsafe_load(f)
    return res

if __name__ == "__main__":
    args = get_args()

    save_dir = "/home/fi5666wi/Python/Brain-CT/saved_models"
    model_name = args.model_name
    model_path = os.path.join(save_dir, model_name, f"version_{args.version}")

    with open(os.path.join(model_path, 'config.yaml'), "r") as f:
        config = yaml.safe_load(f)
        config = Namespace(**config)

    if args.model == "genunet":
        boot = volume_bootstrap(config, model_path, gen=True)
    else:
        boot = volume_bootstrap(config, model_path)
    #boot = load_results(model_path)

    stats = statistics(boot)
    print(stats["Dice"])

    with open(os.path.join(model_path, "volume_bootstrap.yaml"), "w") as f:
        yaml.dump(boot, f)

    save_as_csv(stats, os.path.join(model_path, "vol_stats.csv"))