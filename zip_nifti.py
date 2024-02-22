import SimpleITK as sitk
import os

def zip_nifti(path):
    """
    Zip all nifti files in a folder
    :param path: path to folder containing nifti files
    :return: None
    """
    files = os.listdir(path)
    for file in files:
        if file.endswith(".nii"):
            img = sitk.ReadImage(os.path.join(path, file))
            sitk.WriteImage(img, os.path.join(path, file + ".gz"))
            os.remove(os.path.join(path, file))

if __name__ == "__main__":
    """
    zip_nifti("/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2024-01-15_v3")
    zip_nifti("/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2024-01-15_v4")
    zip_nifti("/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2024-01-16_v1")
    zip_nifti("/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2024-01-16_v4")
    zip_nifti("/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2024-01-23_v2")
    zip_nifti("/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2024-01-23_v4")
    zip_nifti("/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2024-01-24_v1")
    zip_nifti("/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2024-01-24_v4")
    """
    zip_nifti("/home/fi5666wi/Brain_CT_MR_data/OUT/CT")
