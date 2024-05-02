import numpy as np
from scipy import stats
import os
from faster_bootstrap import load_results, bootstrap

def welch_ttest(x, y):
    n_classes = x.shape[1]
    t = np.zeros(n_classes)
    p = np.zeros(n_classes)
    for i in range(n_classes):
        t[i], p[i] = stats.ttest_ind(x[:, i], y[:, i], equal_var=False)
    return t, p


def get_bootstrap(mod):
    res = load_results(mod)
    return bootstrap(res)

if __name__ == "__main__":
    save_dir = "/home/fi5666wi/Python/Brain-CT/saved_models"
    model_a = os.path.join(save_dir, "unet_plus_plus_3d_2024-04-08", f"version_0")
    model_b = os.path.join(save_dir, "unet_plus_plus_3d_2024-04-05", f"version_0")

    bs_a = get_bootstrap(model_a)
    bs_b = get_bootstrap(model_b)

    dice_t, dice_p = welch_ttest(bs_a['Dice'], bs_b['Dice'])
    print(f"Dice: t = {dice_t}, p = {dice_p}")

    iou_t, iou_p = welch_ttest(bs_a['IoU'], bs_b['IoU'])
    print(f"IoU: t = {iou_t}, p = {iou_p}")


