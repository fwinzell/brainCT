import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import Dataset2hD, BrainDataset
from scipy.stats import norm


if __name__ == "__main__":
    datafolder = "/home/fi5666wi/Brain_CT_MR_data/DL"
    energies = [50, 70, 120]

    #fig = plt.figure(0)
    means = np.zeros((3,3))
    for j, level in enumerate(energies):
        tr_cases = [(f"{datafolder}/11_Lh96_M{level}_l_T1.nii", f"{datafolder}/11_Lh96_seg3.nii")]
        """
            [(f"{datafolder}/Mc43_M{level}_l_T1.nii", f"{datafolder}/Mc43_seg3.nii")]
            [(f"{datafolder}/Mbr57_M{level}_l_T1.nii", f"{datafolder}/Mbr57_seg3.nii")] + \
            [(f"{datafolder}/Kg40_M{level}_l_T1.nii", f"{datafolder}/Kg40_seg3.nii")] + \
            [(f"{datafolder}/8_Ms59_M{level}_l_T1.nii", f"{datafolder}/8_Ms59_seg3.nii")] + \
            
        """

        dataset = BrainDataset(tr_cases)

        GM = []
        WM = []
        CSF = []
        for data in dataset:
            img = data["img"][1]
            seg = data["seg"]
            if np.count_nonzero(seg) == 0:
                continue
            # img[:, np.nonzero(seg[0, :, :])[0], np.nonzero(seg[0, :, :])[1]].
            csf = img[np.nonzero(seg[2, :, :])[0], np.nonzero(seg[2, :, :])[1]].flatten()
            gm = img[np.nonzero(seg[1, :, :])[0], np.nonzero(seg[1, :, :])[1]].flatten()
            wm = img[np.nonzero(seg[0, :, :])[0], np.nonzero(seg[0, :, :])[1]].flatten()

            CSF += list(csf)
            GM += list(gm)
            WM += list(wm)


        #X_gm = np.ones(len(GM))*level
        #X_wm = np.ones(len(WM))*level
        #plt.plot(X_gm, GM, 'rx')
        #plt.plot(X_wm, WM, 'gx')
        GM, WM, CSF = np.array(GM), np.array(WM), np.array(CSF)
        means[2, j] = np.mean(CSF)
        means[1, j] = np.mean(GM)
        means[0, j] = np.mean(WM)

        x_lim_csf = (np.min(CSF), np.max(CSF))
        X_csf = np.linspace(x_lim_csf[0], x_lim_csf[1], 200)
        csf_pdf = norm.pdf(X_csf, means[2, j], np.std(CSF))

        x_lim_gm = (np.min(GM), np.max(GM))
        X_gm = np.linspace(x_lim_gm[0], x_lim_gm[1], 200)
        gm_pdf = norm.pdf(X_gm, means[1, j], np.std(GM))

        x_lim_wm = (np.min(WM), np.max(WM))
        X_wm = np.linspace(x_lim_wm[0], x_lim_wm[1], 200)
        wm_pdf = norm.pdf(X_wm, means[0, j], np.std(WM))

        plt.figure(level)
        plt.plot(X_gm, gm_pdf, 'r', linewidth=2)
        plt.plot(X_wm, wm_pdf, 'g', linewidth=2)
        plt.plot(X_csf, csf_pdf, 'b', linewidth=2)

    plt.figure(1)
    plt.plot(energies, means[2, :], 'bo-', markerfacecolor="black")
    plt.plot(energies, means[1, :], 'ro-', markerfacecolor="black")
    plt.plot(energies, means[0, :], 'go-', markerfacecolor="black")
    plt.show()








