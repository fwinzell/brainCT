import os
import numpy as np
import torch
from time import process_time, time
from datetime import timedelta
import pandas as pd

from monai.transforms import AsDiscrete
from brainCT.inference.faster_bootstrap import get_dataset
from brainCT.main import get_model
import SimpleITK as sitk

from brainCT.inference.ensemble import calculate_metrics, get_config

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("running on: {}".format(device))

def eval_cross_validation(cv_dir, test_IDs, config, save=False, save_name="model", batch_size=16, prob_map=False):
    binarize = AsDiscrete(threshold=0.5)

    results = {'Model': [], 
               'Volume': [],
               'Dice_WM': [],
               'Dice_GM': [],
               'Dice_CSF': [],
               'Time': [],
               'CPU Time': []}

    t_start = process_time()
    real_time_start = time()

    for f in os.listdir(cv_dir):
        print(f"Loading model {f}...")
        model_dir = os.path.join(cv_dir, f)
        config = get_config(model_dir)
        model = get_model(config)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'last.pth')), strict=True)
        model.eval()
        #ensemble.append(model)

        for cid in test_IDs:
            print(f"Evaluating {cid}...")
            cpu_vol_start = process_time()
            t_vol_start = time()

            dataset = get_dataset(config, [cid], i3d=config.use_3d_input) 
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
            
            out_vol = np.zeros((config.n_classes, len(dataset), 256, 256))
            gt_map = np.zeros((config.n_classes, len(dataset), 256, 256))
            for k, batch in enumerate(loader):
                if config.use_3d_input:
                    input = torch.stack([batch["img_50"], batch["img_70"], batch["img_120"]], dim=1)
                    label = batch["seg"]
                else:
                    input, label = (batch["img"], batch["seg"])


                with torch.no_grad():   
                    output = model(input)
                    if type(output) == list:
                        output = output[0]

                    if config.sigmoid:
                        y_pred = torch.sigmoid(output)
                    else:
                        y_pred = output

                    if not prob_map:
                        y_pred = binarize(y_pred)

                    end_slice = min((k+1) * batch_size, len(dataset))
                    start_slice = k * batch_size
                    
                    out_vol[:, start_slice:end_slice, :, :] = np.moveaxis(y_pred.cpu().numpy(), 0, 1)
                    gt_map[:, start_slice:end_slice, :, :] = np.moveaxis(label.cpu().numpy(), 0, 1)

            t_vol_end = time()
            cpu_vol_end = process_time()
            t_vol_elapsed = t_vol_end - t_vol_start
            print(f"Time for {cid}: {np.round(t_vol_elapsed, 1)}s")


            if prob_map:
                dice_scores = calculate_metrics(binarize(out_vol), gt_map)['Dice']
            else:
                dice_scores = calculate_metrics(out_vol, gt_map)['Dice']

            results['Model'].append(f)
            results['Volume'].append(cid)
            results['Dice_WM'].append(dice_scores[0])
            results['Dice_GM'].append(dice_scores[1])
            results['Dice_CSF'].append(dice_scores[2])
            results['Time'].append(np.round(t_vol_elapsed, 1))
            results['CPU Time'].append(np.round(cpu_vol_end - cpu_vol_start, 1))

            if save:
                if prob_map:
                    save_dir = os.path.join(save_name, f"{f}_prob")
                else:
                    save_dir = os.path.join(save_name, f)

                save_output(save_dir, out_vol, cid)
                #print(f"Segmentation saved as {save_dir}.nii.gz")

    t_end = process_time()
    real_time_end = time()
    cpu_elapsed = np.round(t_end - t_start)
    print(f"Total Elapsed CPU Time: {timedelta(seconds=cpu_elapsed)}")
    print(f"Total Elapsed Real Time: {timedelta(seconds=np.round(real_time_end - real_time_start))}")
    return results

def save_output(save_dir, out_vol, test_case):
    dir_name = os.path.join("/home/fi5666wi/Brain_CT_MR_data/OUT/", save_dir)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    sitk.WriteImage(sitk.GetImageFromArray(out_vol), os.path.join(dir_name, f"{test_case}_seg.nii.gz"), imageIO="NiftiImageIO")
    print("Saved output @ ", os.path.join(dir_name, f"{test_case}_seg.nii.gz"))



if __name__ == "__main__":
    save_dir = "/home/fi5666wi/Python/Brain-CT/saved_models"
    model_names = ["crossval_2025-01-27", "crossval_2025-01-31", "crossval_2025-01-30", "crossval_2025-01-21", "crossval_2025-01-23"]
    archs = ["unet_3d_0", "unet_0", "unet_att_0", "unet_plus_plus_0", "unet_plus_plus_baseline"]

    for i in range(len(model_names)):
        model_name = model_names[i]
        arch = archs[i]
        cv_dir = os.path.join(save_dir, model_name, arch)

        config = get_config(os.path.join(cv_dir, 'version_0'))

        # Removed due to insufficient quality on MRI imageE
        # 1_BN52, 2_CK79, 3_CL44, 4_JK77, 6_MBR57, 12_AA64, 29_MS42
        test_IDs = ["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"]
        IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
            "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
            "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"]  # 3mm

        results = eval_cross_validation(cv_dir, test_IDs, config, save=True, save_name=model_name, prob_map=True)

        print(f"Average Dice WM: {np.around(np.mean(results['Dice_WM']), decimals=4)}")
        print(f"Average Dice GM: {np.around(np.mean(results['Dice_GM']), decimals=4)}")
        print(f"Average Dice CSF: {np.around(np.mean(results['Dice_CSF']), decimals=4)}")
        
        res_df = pd.DataFrame(results)
        res_df.to_csv(f"/home/fi5666wi/Brain_CT_MR_data/crossval_results/{model_name}.csv")
