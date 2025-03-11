import numpy as np
from scipy import stats
import os
import pandas as pd
from faster_bootstrap import bootstrap, load_bootstrap
from bootstrap_volumes import load_results, volume_bootstrap
from statsmodels.multivariate.manova import MANOVA


def welch_ttest(x, y):
    n_classes = x.shape[1]
    t = np.zeros(n_classes)
    p = np.zeros(n_classes)
    for i in range(n_classes):
        t[i], p[i] = stats.ttest_ind(x[:, i], y[:, i], equal_var=False)
    return t, p


def get_bootstrap(mod):
    res = load_bootstrap(mod)
    return bootstrap(res)


def get_volume_bootstrap(mod):
    res = load_results(mod)
    return res

def print_significance(ps):
    sign = []
    for p in ps:
        if p < 0.001:
            sign.append("***")
        elif p < 0.01:
            sign.append("**")
        elif p < 0.05:
            sign.append("*")
        elif p < 0.1:
            sign.append(".")
        else:
            sign.append("None")
    print(sign)


def manova_test(bs_a, bs_b, n=6, model_names=None):
    if model_names is None:
        model_names = ['a', 'b']
    data = {
        'model': [model_names[0]] * 3 * n + [model_names[1]] * 3 * n,
        'class': ['WM', 'GM', 'CSF'] * 2 * n,
        'Dice': np.concatenate((bs_a['Dice'].flatten(), bs_b['Dice'].flatten()), axis=0),
        'IoU': np.concatenate((bs_a['IoU'].flatten(), bs_b['IoU'].flatten()), axis=0),
    }
    df = pd.DataFrame(data)
    # check for normality
    print("Shapiro test for Dice", stats.shapiro(df['Dice']))
    print("Shapiro test for IoU", stats.shapiro(df['IoU']))

    maov = MANOVA.from_formula('Dice + IoU ~ model * class', data=df)
    print(maov.mv_test())
    return maov


if __name__ == "__main__":
    save_dir = "/home/fi5666wi/Python/Brain-CT/saved_models"
    models = {
        "baseline": os.path.join(save_dir, "unet_plus_plus_2024-04-02", "version_0"),
        "unet_plus_aug": os.path.join(save_dir, "unet_plus_plus_2024-04-02", "version_1"),
        "unet_aug": os.path.join(save_dir, "unet_2024-04-03", "version_0"),
        "unet_plus_fuse": os.path.join(save_dir, "unet_plus_plus_3d_2024-04-05", "version_0"),
        "unet_gated": os.path.join(save_dir, "unet_att_2024-04-03", "version_0"),
        "unet_fuse": os.path.join(save_dir, "unet_2024-04-03", "version_1"),
        "gen_unet": os.path.join(save_dir, "gen_2024-04-10", "version_0")
    }

    for i in range(0, 7):
        if i == 1:
            continue
        bs_a = get_volume_bootstrap(models['unet_plus_aug'])
        model_name = list(models.keys())[i]
        bs_b = get_volume_bootstrap(models[model_name])

        print(f"Comparing {model_name} with unet_plus_aug")
        dice_t, dice_p = welch_ttest(bs_a['Dice'], bs_b['Dice'])
        print(f"Dice: t = {dice_t}, p = {np.round(dice_p, 5)}")
        print_significance(dice_p)

        iou_t, iou_p = welch_ttest(bs_a['IoU'], bs_b['IoU'])
        print(f"IoU: t = {iou_t}, p = {np.round(iou_p, 5)}")
        print_significance(iou_p)

        # manova_test(bs_a, bs_b, model_names=['baseline', 'unet_plus_aug'])
