import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import Dataset2hD, BrainDataset
from scipy.stats import norm
from prettytable import PrettyTable


if __name__ == "__main__":
    datafolder = "/home/fi5666wi/Brain_CT_MR_data/DL"
    energies = [50, 70, 120]

    IDs = ["1_Bn52", "4_Jk77", "5_Kg40", "6_Mbr57", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39"]

    means = np.zeros((3, len(energies)))
    stds = np.zeros((3, len(energies)))

    for j, level in enumerate(energies):
        tr_cases = [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[3:]]

        dataset = BrainDataset(tr_cases)

        GM = []
        WM = []
        CSF = []
        for data in dataset:
            img = data["img"][1]
            seg = data["seg"]
            if np.count_nonzero(seg) == 0:
                continue
            csf = img[np.nonzero(seg[2, :, :])[0], np.nonzero(seg[2, :, :])[1]].flatten()
            gm = img[np.nonzero(seg[1, :, :])[0], np.nonzero(seg[1, :, :])[1]].flatten()
            wm = img[np.nonzero(seg[0, :, :])[0], np.nonzero(seg[0, :, :])[1]].flatten()

            CSF += list(csf)
            GM += list(gm)
            WM += list(wm)

        GM, WM, CSF = np.array(GM), np.array(WM), np.array(CSF)
        means[2, j] = np.mean(CSF)
        means[1, j] = np.mean(GM)
        means[0, j] = np.mean(WM)

        stds[2, j] = np.std(CSF)
        stds[1, j] = np.std(GM)
        stds[0, j] = np.std(WM)

    table = PrettyTable()
    table.field_names = ["Tissue", "50 kV", "70 kV", "120 kV"]
    table.add_row(["WM", f"{means[0, 0]:.2f} +/- {stds[0, 0]:.2f}", f"{means[0, 1]:.2f} +/- {stds[0, 1]:.2f}", f"{means[0, 2]:.2f} +/- {stds[0, 2]:.2f}"])
    table.add_row(["GM", f"{means[1, 0]:.2f} +/- {stds[1, 0]:.2f}", f"{means[1, 1]:.2f} +/- {stds[1, 1]:.2f}", f"{means[1, 2]:.2f} +/- {stds[1, 2]:.2f}"])
    table.add_row(["CSF", f"{means[2, 0]:.2f} +/- {stds[2, 0]:.2f}", f"{means[2, 1]:.2f} +/- {stds[2, 1]:.2f}", f"{means[2, 2]:.2f} +/- {stds[2, 2]:.2f}"])
    print(table)













