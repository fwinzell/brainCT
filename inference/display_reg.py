from brainCT.inference.display_gui import SegmentationGUI2, SegmentationGUI3, load_nii, load_all_nii, load_gt_nii
import numpy as np
import tkinter as tk
import glob
from brainCT.base.regutils import simpleelastix_utils as sutl
from brainCT.base.dicomutils import dicomutils

def read_raw_mri_from_dicom(vol_name, pf):
    mri_dicom = dicomutils.dicomscan(glob.glob(f"{pf}/DCM/{vol_name}/MR")[0])
    mri_im = sutl.ReadImage(list(mri_dicom.values())[0].sitkimage()[0])
    print(f"Raw MRI shape from DICOM: {mri_im.arr.shape}")

if __name__ == "__main__":
    
    # test "8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"
    # train "5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
    #       "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
    #  3mm  "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"  

    #path = "/home/fi5666wi/Brain_CT_MR_data/matlab/8_Ms59/c01_1_00001_temp_8_Ms59_M70_l_T1_CTseg.nii"
    vol_name = "9_Kh43"
    path = f"/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2025-01-31/version_0_prob/{vol_name}_seg.nii.gz"
    gt_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_seg3.nii"
    ct_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_M70_l_T1.nii"
    mr_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_T1.nii"

    pf = "/home/fi5666wi/Brain_CT_MR_data/LUND_2023_04"
    raw_mr_path = f"{pf}/ANA/{vol_name}/rawavg.nii"
    raw_seg_path = f"{pf}/ANA/{vol_name}/aseg.nii"

    seg = load_nii(path, prob_map=True)
    aseg = load_nii(raw_seg_path, prob_map=False)
    raw_mri = load_nii(raw_mr_path, prob_map=False)

    #read_raw_mri_from_dicom(vol_name, pf)

    cncct_dicom = dicomutils.dicomscan(glob.glob(f"{pf}/DCM/{vol_name}/CT/70keV")[0])
    cNCCT = sutl.ReadImage(list(cncct_dicom.values())[0].sitkimage()[0])
    # clamp to 100 range for reg and downstream proc as well
    cNCCT.arr[cNCCT.arr < 0] = 0
    cNCCT.arr[cNCCT.arr > 100] = 100
    cNCCT.update_arr()
        
    _, valid_slices = load_gt_nii(gt_path)
    mri, ct, gt = load_all_nii(mr_path, ct_path, gt_path, using_3d=False)

    if seg.shape[1] == gt.shape[1]+2:
        seg = seg[:, 1:-1, :, :]  # remove first and last slice in 2.5D
    #raw_mri = np.expand_dims(raw_mri[valid_slices, :, :], 0)
    #raw_mri = raw_mri + np.abs(np.min(raw_mri))
    #raw_mri = raw_mri / np.max(raw_mri)
    #raw_mri = np.concatenate((raw_mri, raw_mri, raw_mri), axis=0)


    print("Processed data shapes:")
    print(seg.shape)
    print(gt.shape)
    print(mri.shape)
    print(ct.shape)

    print("Raw data shapes:")
    print(aseg.shape)
    print(raw_mri.shape)

    print("Raw CT shape:")
    print(cNCCT.arr.shape)

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Image Viewer")

    # Create an instance of the ImageApp class
    app = SegmentationGUI2(root, seg, gt, mri, ct)
    #app = SegmentationGUI3(root, gt, mri, ct)

    # Run the Tkinter event loop
    root.mainloop()

    