
from display_gui import load_all_nii, load_nii
import matplotlib.pyplot as plt
import numpy as np

from skimage import metrics

def visualize_hausdorff(seg, gt):
    (wm, gm, csf) = np.split(seg, 3, axis=0)
    (wm_t, gm_t, csf_t) = np.split(gt, 3, axis=0)

    hsd = metrics.hausdorff_distance(csf, csf_t)
    hausdorff_point_a, hausdorff_point_b = metrics.hausdorff_pair(csf, csf_t)
    print(hsd)
    print(hausdorff_point_a)
    print(hausdorff_point_b)

if __name__ == "__main__":
    
    # test "8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"
    # train "5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
    #       "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
    #  3mm  "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"  

    #path = "/home/fi5666wi/Brain_CT_MR_data/matlab/8_Ms59/c01_1_00001_temp_8_Ms59_M70_l_T1_CTseg.nii"
    vol_name = "19_LH64"
    path = f"/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2025-01-31/version_0_prob/{vol_name}_seg.nii.gz"
    gt_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_seg3.nii"
    ct_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_M70_l_T1.nii"
    mr_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_T1.nii"
    seg = load_nii(path, prob_map=True)
    #gt = load_gt_nii(gt_path)
    mri, ct, gt = load_all_nii(mr_path, ct_path, gt_path, using_3d=False)
    print(seg.shape)
    print(gt.shape)

    visualize_hausdorff(seg, gt)