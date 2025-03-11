import cv2
import numpy as np

def display_result(y_pred, label, wait=100, n_classes=3):
    # prediction
    seg = y_pred.detach().cpu().numpy()
    batch_size = seg.shape[0]
    for i in range(batch_size):
        gm = seg[i, 1, :, :]
        wm = seg[i, 0, :, :]
        #border = seg[0, 2, :, :]
        if n_classes == 3:
            csf = seg[i, 2, :, :]
        else:
            csf = 0*gm

        wmgm_rgb = np.stack((wm, gm, csf), axis=0) #0 * gm), axis=0)
        img = np.moveaxis(np.uint8(wmgm_rgb * 255), source=0, destination=-1)

        # target
        tar = label.detach().cpu().numpy()
        if tar.ndim == 4:
            tar = tar[i, :, :, :].squeeze()
        if n_classes == 2:
            tar = np.concatenate((tar.squeeze(), np.zeros((1, 256, 256))), axis=0)
        else:
            tar = tar.squeeze()
        tar_img = np.moveaxis(np.uint8(tar * 255), source=0, destination=-1)

        # find all correct and incorrect pixels, not including background and make an image with green and red colors
        pred_img = np.zeros((256, 256, 3), dtype=np.uint8)
        #correct_gm = np.logical_and(gm, tar[1, :, :])
        #correct_wm = np.logical_and(wm, tar[0, :, :])
        #correct_csf = np.logical_and(csf, tar[2, :, :])
        corr = np.logical_and(gm, tar[1, :, :]) + np.logical_and(wm, tar[0, :, :]) + np.logical_and(csf, tar[2, :, :])
        pred_img[corr, :] = [0, 255, 0]
        incorr = np.logical_xor(gm, tar[1, :, :]) + np.logical_xor(wm, tar[0, :, :]) + np.logical_xor(csf, tar[2, :, :])
        pred_img[incorr, :] = [0, 0, 255]

        # ratio of correct pixels
        ratio = np.sum(corr) / (np.sum(corr) + np.sum(incorr))

        # display
        cv2.imshow('Segment', img)
        #cv2.imshow('GM', np.moveaxis(gm*255, source=0, destination=-1))
        #cv2.imshow('WM', np.moveaxis(wm * 255, source=0, destination=-1))
        #cv2.imshow('CSF', np.moveaxis(csf * 255, source=0, destination=-1))
        #cv2.imshow('Border', border)
        cv2.imshow('Target', tar_img)
        cv2.putText(pred_img, f"Ratio: {ratio:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('Correct/incorrect', pred_img)
        cv2.waitKey(wait)


def display_seg_and_recon(y_pred, label, recon, target, wait=100):
    # Segmentation
    seg = y_pred.detach().cpu().numpy()
    gm = seg[0, 1, :, :]
    wm = seg[0, 0, :, :]
    csf = seg[0, 2, :, :]

    wmgm_rgb = np.stack((wm, gm, csf), axis=0) #0 * gm), axis=0)
    img = np.moveaxis(np.uint8(wmgm_rgb * 255), source=0, destination=-1)

    tar = label.detach().cpu().numpy()
    tar = tar.squeeze()
    tar_img = np.moveaxis(np.uint8(tar * 255), source=0, destination=-1)

    # find all correct and incorrect pixels, not including background and make an image with green and red colors
    pred_img = np.zeros((256, 256, 3), dtype=np.uint8)
    corr = np.logical_and(gm, tar[1, :, :]) + np.logical_and(wm, tar[0, :, :]) + np.logical_and(csf, tar[2, :, :])
    pred_img[corr, :] = [0, 255, 0]
    incorr = np.logical_xor(gm, tar[1, :, :]) + np.logical_xor(wm, tar[0, :, :]) + np.logical_xor(csf, tar[2, :, :])
    pred_img[incorr, :] = [0, 0, 255]

    # ratio of correct pixels
    ratio = np.sum(corr) / (np.sum(corr) + np.sum(incorr))

    # Reconstruction
    recon = recon.detach().cpu().numpy()
    recon = np.squeeze(np.uint8(recon * 255))
    target = target.detach().cpu().numpy()
    target = np.squeeze(np.uint8(target * 255))

    # display
    cv2.imshow('Segment', img)
    cv2.imshow('Ground Truth', tar_img)
    cv2.putText(pred_img, f"Ratio: {ratio:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('Correct/incorrect', pred_img)

    cv2.imshow('Reconstruction', recon)
    cv2.imshow('Target', target)
    cv2.waitKey(wait)
