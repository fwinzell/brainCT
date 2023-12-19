import numpy as np
from monai.data import GridPatchDataset, DataLoader, PatchIter, PatchDataset, Dataset
from monai.transforms import RandShiftIntensity, RandSpatialCropSamples, Compose, RandFlipd
import SimpleITK as sitk
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union
from torch.utils.data import Subset
import torch
import collections.abc
from monai.transforms import Compose, Randomizable, Transform, apply_transform, RandRotated, RandAffined, ToTensord, \
    RandGaussianNoised, NormalizeIntensity, ScaleIntensityd, RandGaussianSmoothd, RandAdjustContrastd, RandGibbsNoised, \
    RandZoomd
from torchvision.transforms import GaussianBlur, RandomApply
import cv2


class Dataset2hD(Dataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        img = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(dataurls[0][0])),0)
        indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(dataurls[0][1])),0)
        GM = indxs == 2
        WM = indxs == 1
        seg = np.concatenate( (WM,GM),axis=0)
        for k in dataurls[1:]:
            indxs =  np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[1])),0)
            cimg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[0])), 0)
            valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
            cimg = cimg[:,valid_slices[:,0],:,:]
            indxs = indxs[:,valid_slices[:,0],:,:]
            GM = indxs == 2
            WM = indxs == 1

            cseg = np.concatenate((WM, GM), axis=0)

            img = np.concatenate((img, cimg), axis=1)
            seg = np.concatenate((seg, cseg), axis=1)

        img[img<0] = 0
        img[img > 100] = 100
        images = {"img": (img-30.0)/5.0, "seg": seg.astype(np.uint8)}
        super().__init__( images, transform )

    def __len__(self) -> int:
        return self.data["img"].shape[1]-2

    def _transform(self, index: int):
        #Fetch single data item from `self.data`.

        imgblock = self.data["img"][0,index:index+3,:,:]
        segblock = self.data["seg"][:,index+1,:,:]
        data_i = {"img": imgblock, "seg": segblock}

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i


    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)
        return self._transform(index)


class BrainDataset(Dataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        img = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(dataurls[0][0])), 0)
        indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(dataurls[0][1])), 0)
        CSF = indxs == 3
        GM = indxs == 2
        WM = indxs == 1
        #Border = self.find_border(WM, GM)
        seg = np.concatenate((WM, GM, CSF), axis=0)
        for k in dataurls[1:]:
            indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[1])), 0)
            cimg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[0])), 0)
            valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
            cimg = cimg[:, valid_slices[:, 0], :, :]
            indxs = indxs[:, valid_slices[:, 0], :, :]
            CSF = indxs == 3
            GM = indxs == 2
            WM = indxs == 1
            #Border = self.find_border(WM, GM)

            cseg = np.concatenate((WM, GM, CSF), axis=0)

            img = np.concatenate((img, cimg), axis=1)
            seg = np.concatenate((seg, cseg), axis=1)

        img[img < 0] = 0
        img[img > 100] = 100
        images = {"img": (img - 30.0) / 5.0, "seg": seg.astype(np.uint8)}
        super().__init__(images, transform)

    def __len__(self) -> int:
        return self.data["img"].shape[1] - 2

    def find_border(self, WM, GM):
        assert WM.shape == GM.shape
        img = np.zeros(WM.shape)
        mat = (WM + GM).astype(int)
        for idx in range(WM.shape[1]):
            contour,_ = cv2.findContours(np.moveaxis(mat[:,idx,:,:].astype(np.uint8), 0, -1), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            img[:,idx,:,:] = cv2.drawContours(np.zeros(img.shape[-2:]), contour, -1, color=1, thickness=2)
        return img == 1


    def _transform(self, index: int):
        # Fetch single data item from `self.data`.

        imgblock = self.data["img"][0, index:index + 3, :, :]
        segblock = self.data["seg"][:, index + 1, :, :]
        data_i = {"img": imgblock, "seg": segblock}

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            return Subset(dataset=self, indices=index)
        return self._transform(index)


class SpectralDatasetLight(Dataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        img_50, img_70, img_120, seg = None, None, None, None
        for j,k in enumerate(dataurls):
            indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[3])), 0)
            cimg_50 = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[0])), 0)
            cimg_70 = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[1])), 0)
            cimg_120 = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[2])), 0)

            valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
            cimg_50 = cimg_50[:, valid_slices[:, 0], :, :]
            cimg_70 = cimg_70[:, valid_slices[:, 0], :, :]
            cimg_120 = cimg_120[:, valid_slices[:, 0], :, :]
            indxs = indxs[:, valid_slices[:, 0], :, :]

            CSF = indxs == 3
            GM = indxs == 2
            WM = indxs == 1
            cseg = np.concatenate((WM, GM, CSF), axis=0)

            if j == 0:
                img_50, img_70, img_120 = cimg_50, cimg_70, cimg_120
                seg = cseg
            else:
                img_50 = np.concatenate((img_50, cimg_50), axis=1)
                img_70 = np.concatenate((img_70, cimg_70), axis=1)
                img_120 = np.concatenate((img_120, cimg_120), axis=1)
                seg = np.concatenate((seg, cseg), axis=1)

        img_50[img_50 < 0] = 0
        img_50[img_50 > 100] = 100
        img_70[img_70 < 0] = 0
        img_70[img_70 > 100] = 100
        img_120[img_120 < 0] = 0
        img_120[img_120 > 100] = 100
        images = {"img_50": (img_50 - 30.0) / 5.0,
                  "img_70": (img_70 - 30.0) / 5.0,
                  "img_120": (img_120 - 30.0) / 5.0,
                  "seg": seg.astype(np.uint8)}
        super().__init__(images, transform)

    def __len__(self) -> int:
        return self.data["img_50"].shape[1]

    def __getitem__(self, index: int):
        # Fetch single data item from `self.data`.

        img50 = np.expand_dims(self.data["img_50"][0, index, :, :], 0)
        img70 = np.expand_dims(self.data["img_70"][0, index, :, :], 0)
        img120 = np.expand_dims(self.data["img_120"][0, index, :, :], 0)
        imgblock = np.concatenate((img50, img70, img120), axis=0)
        segblock = self.data["seg"][:, index, :, :]
        data_i = {"img": imgblock, "seg": segblock}

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i


class BrainXLDataset(Dataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        self.urls = dataurls
        self.transform = transform
        self.slice_idxs = self._preprocess()
        self.data = None
        self.cached = -1

    def __len__(self):
        return self.slice_idxs[-1]

    def _preprocess(self):
        # Need to handle 3mm CT images here as well?
        n_slices = [0]
        for f in self.urls:
            indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(f[-1])), 0)
            # cimg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[0])), 0)
            valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
            n_slices.append(valid_slices.shape[0] - 2) # -2 to account for last and first slice in 2.5D
        return np.cumsum(n_slices)

    def _load_data(self, url):
        indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[1])), 0)
        img = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[0])), 0)
        valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
        img = img[:, valid_slices[:, 0], :, :]
        indxs = indxs[:, valid_slices[:, 0], :, :]
        CSF = indxs == 3
        GM = indxs == 2
        WM = indxs == 1

        seg = np.concatenate((WM, GM, CSF), axis=0)
        img[img < 0] = 0
        img[img > 100] = 100
        self.data = {"img": (img - 30.0) / 5.0, "seg": seg.astype(np.uint8)}

    def _get_vol_idx(self, index):
        # Return index of volume corresponding with index
        return np.min(np.where(self.slice_idxs > index))-1

    def _transform(self, index: int):
        # Fetch single data item from `self.data`.

        imgblock = self.data["img"][0, index:index + 3, :, :]
        segblock = self.data["seg"][:, index + 1, :, :]
        data_i = {"img": imgblock, "seg": segblock}

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i


    def __getitem__(self, index: int):
        vol_idx = self._get_vol_idx(index)
        # Check that the current volume is correct, otherwise update
        if vol_idx != self.cached:
            self._load_data(self.urls[vol_idx])
            self.cached = vol_idx
        sind = index-self.slice_idxs[vol_idx]
        return self._transform(sind)

class SpectralDataset(BrainXLDataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        super().__init__(dataurls, transform)

    def _preprocess(self):
        # Need to handle 3mm CT images here as well?
        n_slices = [0]
        for f in self.urls:
            indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(f[-1])), 0)
            # cimg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[0])), 0)
            valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
            n_slices.append(valid_slices.shape[0]) # all slices are valid
        return np.cumsum(n_slices)

    def _load_data(self, url):
        indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[-1])), 0)
        valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
        indxs = indxs[:, valid_slices[:, 0], :, :]

        CSF = indxs == 3
        GM = indxs == 2
        WM = indxs == 1
        seg = np.concatenate((WM, GM, CSF), axis=0)

        imgs = []
        for i in range(3):
            img = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[i])), 0)
            img = img[:, valid_slices[:, 0], :, :]
            img[img < 0] = 0
            img[img > 100] = 100
            imgs.append((img - 30.0) / 5.0)

        self.data = {"img_50": imgs[0], "img_70": imgs[1], "img_120": imgs[2], "seg": seg.astype(np.uint8)}

    def _transform(self, index: int):
        imgblock = np.concatenate((np.expand_dims(self.data["img_50"][0, index, :, :], 0),
                                   np.expand_dims(self.data["img_70"][0, index, :, :], 0),
                                   np.expand_dims(self.data["img_120"][0, index, :, :],0))
                                  , axis=0)
        segblock = self.data["seg"][:, index, :, :]
        data_i = {"img": imgblock, "seg": segblock}

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

    def __getitem__(self, index: int):
        vol_idx = self._get_vol_idx(index)
        # Check that the current volume is correct, otherwise update
        if vol_idx != self.cached:
            self._load_data(self.urls[vol_idx])
            self.cached = vol_idx
        sind = index-self.slice_idxs[vol_idx]
        return self._transform(sind)


class VotingDataset(SpectralDataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        super().__init__(dataurls, transform)

    def _preprocess(self):
        n_slices = [0]
        for f in self.urls:
            indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(f[-1])), 0)
            # cimg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[0])), 0)
            valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
            n_slices.append(valid_slices.shape[0] - 2) # -2 to account for last and first slice in 2.5D
        return np.cumsum(n_slices)

    def _transform(self, index: int):
        # Fetch single data item from `self.data`.

        imgblock_50 = self.data["img_50"][0, index:index + 3, :, :]
        imgblock_70 = self.data["img_70"][0, index:index + 3, :, :]
        imgblock_120 = self.data["img_120"][0, index:index + 3, :, :]
        segblock = self.data["seg"][:, index + 1, :, :]
        data_i = {"img_50": imgblock_50, "img_70": imgblock_70, "img_120": imgblock_120,  "seg": segblock}

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i



class HuggingDataset(BrainXLDataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        super().__init__(dataurls, transform)

    def __getitem__(self, index: int):
        vol_idx = self._get_vol_idx(index)
        # Check that the current volume is correct, otherwise update
        if vol_idx != self.cached:
            self._load_data(self.urls[vol_idx])
            self.cached = vol_idx
        sind = index - self.slice_idxs[vol_idx]
        energy = int(self.urls[vol_idx][0].split("_")[-3][1:])
        return self._transform(sind), energy


class WMGMDataset(BrainXLDataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        super().__init__(dataurls, transform)

    def _load_data(self, url):
        indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[1])), 0)
        img = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[0])), 0)
        valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
        img = img[:, valid_slices[:, 0], :, :]
        indxs = indxs[:, valid_slices[:, 0], :, :]
        GM = indxs == 2
        WM = indxs == 1

        seg = WM | GM
        img[img < 0] = 0
        img[img > 100] = 100
        self.data = {"img": (img - 30.0) / 5.0, "seg": seg.astype(np.uint8)}


class MaskDataset(BrainXLDataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        super().__init__(dataurls, transform)

    def _load_data(self, url):
        indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[1])), 0)
        img = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[0])), 0)
        valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
        img = img[:, valid_slices[:, 0], :, :]
        indxs = indxs[:, valid_slices[:, 0], :, :]
        GM = indxs == 2
        WM = indxs == 1
        seg = np.concatenate((WM, GM), axis=0)

        mask = WM | GM
        img = img * mask
        img[img < 0] = 0
        img[img > 100] = 100
        self.data = {"img": (img - 30.0) / 5.0, "seg": seg.astype(np.uint8)}


class InfDataset(SpectralDataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        super().__init__(dataurls, transform)

    def _preprocess(self):
        # Need to handle 3mm CT images here as well?
        n_slices = [0]
        for f in self.urls:
            indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(f[-1])), 0)
            # cimg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[0])), 0)
            valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
            n_slices.append(valid_slices.shape[0])  # all slices are valid
        return np.cumsum(n_slices)

    def _load_data(self, url):
        indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[-1])), 0)
        valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
        indxs = indxs[:, valid_slices[:, 0], :, :]

        CSF = indxs == 3
        GM = indxs == 2
        WM = indxs == 1
        seg = np.concatenate((WM, GM, CSF), axis=0)

        imgs = []
        for i in range(2):
            img = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[i])), 0)
            img = img[:, valid_slices[:, 0], :, :]
            img[img < 0] = 0
            img[img > 100] = 100
            imgs.append((img - 30.0) / 5.0)

        self.data = {"img_70": imgs[0], "img_120": imgs[1], "seg": seg.astype(np.uint8)}

    def _transform(self, index: int):
        # Fetch single data item from `self.data`.

        imgblock_70 = self.data["img_70"][0, index:index + 3, :, :]
        imgblock_120 = self.data["img_120"][0, index:index + 3, :, :]
        segblock = self.data["seg"][:, index + 1, :, :]
        data_i = {"img_70": imgblock_70, "img_120": imgblock_120,  "seg": segblock}

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i


if __name__ == "__main__":
    datafolder = "/home/fi5666wi/Brain_CT_MR_data/DL"

    train_transforms = Compose(
        [RandAffined(keys=["img_50", "img_70", "img_120", "seg"], mode=["bilinear", "bilinear", "bilinear", "nearest"],
                     prob=0.9, shear_range=[0.1, 0.1, 0.1]),
         ToTensord(keys=["img_50", "img_70", "img_120", "seg"]),
         RandGaussianSmoothd(keys=["img_50", "img_70", "img_120"], prob=1.0, sigma_x=(0.2, 2.0), sigma_y=(0.2, 2.0),
                             sigma_z=(0.5, 1.5)),
         RandAdjustContrastd(keys=["img_50", "img_70", "img_120"], prob=1.0, gamma=(0.5, 2.0)),
         RandGibbsNoised(keys=["img_50", "img_70", "img_120"], prob=1.0, alpha=(0, 0.25)),
         RandZoomd(keys=["img_50", "img_70", "img_120", "seg"], mode=["bilinear", "bilinear", "bilinear", "nearest"],
                   prob=1.0, min_zoom=0.9, max_zoom=1.5),
         ScaleIntensityd(keys=["img_50", "img_70", "img_120"], minv=0.0, maxv=1.0)])
    
    energies = [50, 70, 120]
    IDs = ["25_HH57", "26_LB59", "29_MS42", "31_EM88", "32_EN56", "34_LO45"]
    tr_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                for cid in IDs[3:]]

    train_datset = VotingDataset(tr_cases, train_transforms)
    loader = DataLoader(train_datset, batch_size=2, shuffle=False, num_workers=0)
    for i,batte in enumerate(loader):
        img_50 = batte["img_50"].numpy()
        img_70 = batte["img_70"].numpy()
        img_120 = batte["img_120"].numpy()
        seg = batte["seg"].numpy()
        print(f"Loaded {i}")

        if np.count_nonzero(seg) == 0:
            continue
        im50 = np.moveaxis(img_50[0], source=0, destination=-1)
        im70 = np.moveaxis(img_70[0], source=0, destination=-1)
        im120 = np.moveaxis(img_120[0], source=0, destination=-1)
        im50 = np.uint8(((im50-np.min(im50)) / np.max(im50)) * 255)
        im70 = np.uint8(((im70 - np.min(im70)) / np.max(im70)) * 255)
        im120 = np.uint8(((im120 - np.min(im120)) / np.max(im120)) * 255)
        seg = np.moveaxis(np.uint8(seg[0] * 255), source=0, destination=-1)
        cv2.imshow('50keV', im50[:, :, 1])
        cv2.imshow('70keV', im70[:, :, 1])
        cv2.imshow('120keV', im120[:, :, 1])
        cv2.imshow('Target', seg)
        # cv2.imshow('Border', border)
        cv2.waitKey(0)

    """
    training_cases = [(f"{datafolder}/7_Mc43_M50_l_T1.nii", f"{datafolder}/7_Mc43_M70_l_T1.nii", f"{datafolder}/7_Mc43_M120_l_T1.nii", f"{datafolder}/7_Mc43_seg3.nii"),
                      (f"{datafolder}/6_Mbr57_M70_l_T1.nii", f"{datafolder}/6_Mbr57_M70_l_T1.nii", f"{datafolder}/6_Mbr57_M120_l_T1.nii", f"{datafolder}/6_Mbr57_seg3.nii")]
                     # (f"{datafolder}/Ck79_M70_l_T1.nii", f"{datafolder}/Ck79_seg3.nii")]

    train_transforms = Compose([
            #RandAffined(keys=["img", "seg"], mode=["bilinear", "nearest"], prob=0.9, shear_range=[(0.1), (0.1), (0.1)]),
         ToTensord(keys=["img", "seg"]),
         #RandGaussianSmoothd(keys="img", prob=0.5, sigma_x=(0.2, 2.0), sigma_y=(0.2, 2.0), sigma_z=(0.5, 1.5)),
         ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)
        ])

    train_dataset = SpectralDataset(training_cases, train_transforms) #BrainDataset(training_cases, train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    print(len(train_loader))
    for i,batch in enumerate(train_loader):
        img = batch["img"].numpy()
        seg = batch["seg"].numpy()
        #seg = seg[0]
        #border = seg[2,:,:]

        img = np.moveaxis(img[0], source=0, destination=-1)
        print(f"Max: {np.max(img[:,:,1])} Min: {np.min(img[:,:,1])}")
        #seg = np.concatenate((seg[], np.zeros((1, 256, 256))), axis=0)
        seg = np.moveaxis(np.uint8(seg[0] * 255), source=0, destination=-1)
        #border = np.uint8(border * 255)
        #midslice = img[:,:,1] + np.abs(np.min(img[:,:,1])) # middle slice
        #midslice /= np.max(midslice)
        cv2.imshow('50keV', img[:,:,0])
        cv2.imshow('70keV', img[:,:,1])
        cv2.imshow('120keV', img[:,:,2])
        cv2.imshow('Target', seg)
        #cv2.imshow('Border', border)
        cv2.waitKey(500)

    

    energies = [70, 120]

    IDs = ["1_Bn52", "4_Jk77", "5_Kg40", "6_Mbr57", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39"]

    #tr_cases = []
    #for level in energies:
    #    tr_cases += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[3:]]
    wmgm_cases = [(f"{datafolder}/{cid}_M120_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[3:]]

    mask_cases = [(f"{datafolder}/{cid}_M70_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[3:]]

    tr_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                for cid in IDs[3:]]

    train_transforms_1 = Compose(
        [RandAffined(keys=["img", "seg"], mode=["bilinear", "nearest"], prob=0.9, shear_range=[(0.1), (0.1), (0.1)]),
         ToTensord(keys=["img", "seg"]),
         RandGaussianSmoothd(keys="img", prob=0.5, sigma_x=(0.2, 2.0), sigma_y=(0.2, 2.0), sigma_z=(0.5, 1.5)),
         ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)])

    train_transforms_2 = Compose([
        # RandAffined(keys=["img", "seg"], mode=["bilinear", "nearest"], prob=0.9, shear_range=[(0.1), (0.1), (0.1)]),
        ToTensord(keys=["img_70", "img_120", "seg"]),
        # RandGaussianNoised(keys="img", prob=0.5, mean=0.0, std=0.1),
        # RandGaussianSmoothd(keys="img", prob=1.0, sigma_x=(0.1, 2.5), sigma_y=(0.1, 2.5), sigma_z=(1,2))
    ScaleIntensityd(keys=["img_70", "img_120"], minv=0.0, maxv=1.0)])

    train_dataset = MaskDataset(mask_cases, train_transforms_1)
    wmgm_dataset = WMGMDataset(wmgm_cases, train_transforms_1)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    print(len(train_loader))
    for i, batch in enumerate(train_loader):
        img = batch["img"].numpy()
        seg = batch["seg"].numpy()

        img = np.moveaxis(np.uint8(img[:, 0, :, :]*255), source=0, destination=-1)
        #print(f"Max: {np.max(img[:, :, 1])} Min: {np.min(img[:, :, 1])}")
        seg = np.concatenate((seg[0], np.zeros((1, 256, 256))), axis=0)
        seg = np.moveaxis(np.uint8(seg * 255), source=0, destination=-1)
        cv2.imshow('Image', img)
        cv2.imshow('Target', seg)
        # cv2.imshow('Border', border)
        cv2.waitKey(500)

    """


