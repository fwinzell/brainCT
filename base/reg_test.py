from regutils import simpleelastix_utils as sutl
from dicomutils import dicomutils
import SimpleITK as sitk
import glob
import numpy as np
import pandas as pd
import os

labels = pd.read_excel("/home/fi5666wi/Brain_CT_MR_data/DL/labels.xlsx", sheet_name="labels")

targetval = {"WM": 1, "GM": 2, "CSF": 3, "UNDEFINED": 0}

CTtypes = {"M50": "50keV*", "M70": "70keV*", "M120": "120keV*"} #, "conv": "*conv*"}

# IDs = ["8_Ms59",  "9_Kh43",  "10_Ca58",  "11_Lh96" , "12_Aa64"] #,  "Mc43"]
#IDs = ["13_NK51", "17_AL67", "18_MN44", "20_AR94", "22_CM63", "23_SK52", "24_SE39", "25_HH57", "26_LB59", "29_MS42",
#       "31_EM88", "32_EN56", "33_ET51", "34_LO45"]
#IDs = ["25_HH57", "26_LB59", "29_MS42", "31_EM88", "32_EN56", "33_ET51", "34_LO45"]
IDs = ["21_JP42"] #["27_IL48", "30_MJ80"]
for cid in IDs:
    pf = f"/home/fi5666wi/Brain_CT_MR_data/LUND_2023_09"
    T1img = sutl.ReadImage(f"{pf}/ANA/{cid}/rawavg.nii")
    seg = sutl.ReadImage(f"{pf}/ANA/{cid}/aseg.nii")
    brainmask = (T1img.arr.copy() > 0).astype(np.uint8)
    brainmaskimg = sutl.arr2img(brainmask, T1img.img)
    print(f"{pf}/{cid}/MonoE_70keV*")
    NCCT = {}
    for CTtype, search_str in CTtypes.items():
        print(f"Reading {CTtype}")
        cncct_dicom = dicomutils.dicomscan(glob.glob(f"{pf}/DCM/{cid}/CT/{search_str}")[0])
        cNCCT = sutl.ReadImage(list(cncct_dicom.values())[0].sitkimage()[0])
        # clamp to 100 range for reg and downstream proc as well
        cNCCT.arr[cNCCT.arr < 0] = 0
        cNCCT.arr[cNCCT.arr > 100] = 100
        cNCCT.update_arr()
        NCCT[CTtype] = cNCCT

    # now reg M70 and apply to rest
    op = sutl.rigidA2B(NCCT["M70"].img, T1img.img,
                       customxfm='/usr/matematik/fi5666wi/Python/brainCT/base/utils/regutils/MUTUAL4.txt',
                       fixedmask=brainmaskimg)
    sitk.WriteImage(op["moving_l_fixed"], f"{pf}/DL/{cid}_M70_l_T1.nii")

    sitk.WriteImage(T1img.img, f"{pf}/DL/{cid}_T1.nii")

    for ctres in ["M50", "M120"]: #, "conv"]:
        CTresampled = sutl.resample(NCCT[ctres].img, op["xfmmaps"])
        sitk.WriteImage(CTresampled, f"{pf}/DL/{cid}_{ctres}_l_T1.nii")
    # apply to other channels too

    # sitk.WriteImage(NCCT.img,'NCCT.nii')

    # collapse labels for GM and WM
    for indx, typ in zip(labels.label.values, labels.TYPE.values):
        seg.arr[seg.arr == indx] = targetval[typ]

    seg.update_arr()

    sitk.WriteImage(seg.img, f"{pf}/DL/{cid}_seg3.nii")
