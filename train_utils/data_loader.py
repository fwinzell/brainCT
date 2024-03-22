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


class SimpleBrainDataset(Dataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        self.urls = dataurls
        self.transform = transform
        self.data = None
        self.cached = -1

    def __len__(self):
        return len(self.urls)*256

    def _load_data(self, url):
        indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[1])), 0)
        img = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[0])), 0)

        CSF = indxs == 3
        GM = indxs == 2
        WM = indxs == 1

        seg = np.concatenate((WM, GM, CSF), axis=0)
        img[img < 0] = 0
        img[img > 100] = 100
        self.data = {"img": (img - 30.0) / 5.0, "seg": seg.astype(np.uint8)}


    def _transform(self, index: int):
        # Fetch single data item from `self.data`.

        imgblock = self.data["img"][:, index, :, :]
        segblock = self.data["seg"][:, index, :, :]
        data_i = {"img": imgblock, "seg": segblock}

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

    def __getitem__(self, index: int):
        vol_idx = int(np.floor(index / 256))
        # Check that the current volume is correct, otherwise update
        if vol_idx != self.cached:
            self._load_data(self.urls[vol_idx])
            self.cached = vol_idx
        sind = index - vol_idx*256
        return self._transform(sind)



class BrainXLDataset(Dataset):
    def __init__(self,
                 dataurls: Sequence,
                 transform: Optional[Callable] = None,
                 n_pseudo: int = 3) -> None:
        self.urls = dataurls
        self.transform = transform
        self.slice_idxs = self._preprocess()
        self.data = None
        self.cached = -1
        self.n_pseudo = n_pseudo if n_pseudo != 0 else 1

    def __len__(self):
        return self.slice_idxs[-1]

    def _preprocess(self):
        # Need to handle 3mm CT images here as well?
        n_slices = [0]
        for f in self.urls:
            indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(f[-1])), 0)
            # cimg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[0])), 0)
            valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
            n_slices.append(valid_slices.shape[0]) # EDIT -2 not needed anymore? -2 to account for last and first slice in 2.5D
        return np.cumsum(n_slices)

    def _load_data(self, url):
        indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[1])), 0)
        img = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[0])), 0)
        valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
        valid_slices = np.sort(valid_slices, axis=0)
        k = np.floor(self.n_pseudo / 2).astype(int)
        if k > 0:
            valid_slices = np.insert(valid_slices, 0, valid_slices[:k] - k, axis=0)
            valid_slices = np.insert(valid_slices, valid_slices.shape[0], valid_slices[-k:] + k, axis=0)
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

        k = np.floor(self.n_pseudo / 2).astype(int)
        imgblock = self.data["img"][0, index:index + self.n_pseudo, :, :]
        segblock = self.data["seg"][:, index + k, :, :]
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

class BasicBrainDataset(BrainXLDataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        super().__init__(dataurls, transform)

    def _preprocess(self):
        n_slices = [0]
        for f in self.urls:
            indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(f[-1])), 0)
            valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
            n_slices.append(valid_slices.shape[0])
        return np.cumsum(n_slices)

    def _transform(self, index: int):
        imgblock = self.data["img"][:, index, :, :]
        segblock = self.data["seg"][:, index, :, :]
        data_i = {"img": imgblock, "seg": segblock}

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i


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


class ConcatDataset(VotingDataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        super().__init__(dataurls, transform)

    def _transform(self, index: int):
        # Fetch single data item from `self.data`.

        imgblock_50 = self.data["img_50"][0, index:index + 3, :, :]
        imgblock_70 = self.data["img_70"][0, index:index + 3, :, :]
        imgblock_120 = self.data["img_120"][0, index:index + 3, :, :]
        imgblock = np.concatenate((imgblock_50, imgblock_70, imgblock_120), axis=0)
        segblock = self.data["seg"][:, index + 1, :, :]
        data_i = {"img": imgblock, "seg": segblock}

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i



class MultiModalDataset(VotingDataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        super().__init__(dataurls, transform)

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

        mri = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(url[3])), 0)
        mri = mri[:, valid_slices[:, 0], :, :]

        self.data = {"img_50": imgs[0], "img_70": imgs[1], "img_120": imgs[2], "mri": mri, "seg": seg.astype(np.uint8)}

    def _transform(self, index: int):
        # Fetch single data item from `self.data`.

        imgblock_50 = self.data["img_50"][0, index:index + 3, :, :]
        imgblock_70 = self.data["img_70"][0, index:index + 3, :, :]
        imgblock_120 = self.data["img_120"][0, index:index + 3, :, :]
        mriblock = self.data["mri"][:, index + 1, :, :]
        segblock = self.data["seg"][:, index + 1, :, :]
        data_i = {"img_50": imgblock_50, "img_70": imgblock_70, "img_120": imgblock_120,
                  "mri": mriblock, "seg": segblock}

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i


class BootstrappedDataset(BrainXLDataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        super().__init__(dataurls, transform)
        self.samples = range(len(self))
        self.bootstrap = self.samples

    def resample(self):
        self.bootstrap = np.random.choice(self.samples, len(self.samples), replace=True)
        # Optional: Sort the bootstrap samples for speed up
        self.bootstrap = np.sort(self.bootstrap)

    def __getitem__(self, index: int):
        sample = self.bootstrap[index]
        vol_idx = self._get_vol_idx(sample)
        # Check that the current volume is correct, otherwise update
        if vol_idx != self.cached:
            self._load_data(self.urls[vol_idx])
            self.cached = vol_idx
        sind = sample - self.slice_idxs[vol_idx]
        return self._transform(sind)


class BootstrappedDatasetV3(VotingDataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        super().__init__(dataurls, transform)
        self.samples = range(len(self))
        self.bootstrap = self.samples

    def resample(self):
        self.bootstrap = np.random.choice(self.samples, len(self.samples), replace=True)
        # Optional: Sort the bootstrap samples for speed up
        self.bootstrap = np.sort(self.bootstrap)

    def __getitem__(self, index: int):
        sample = self.bootstrap[index]
        vol_idx = self._get_vol_idx(sample)
        # Check that the current volume is correct, otherwise update
        if vol_idx != self.cached:
            self._load_data(self.urls[vol_idx])
            self.cached = vol_idx
        sind = sample - self.slice_idxs[vol_idx]
        return self._transform(sind)


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

    """
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
    """
    train_transforms = Compose([
        ToTensord(keys=["img_50", "img_70", "img_120", "mri", "seg"]),
        ScaleIntensityd(keys=["img_50", "img_70", "img_120"], minv=0.0, maxv=1.0)])
    
    energies = [50, 70, 120]
    IDs = ["25_HH57", "26_LB59", "29_MS42", "31_EM88", "32_EN56", "34_LO45"]
    tr_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies]
                + [f"{datafolder}/{cid}_T1.nii", f"{datafolder}/{cid}_seg3.nii"]
                for cid in IDs[3:]]

    one_case_per_seg = []
    for level in energies:
        one_case_per_seg += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[3:]]

    #brainset = SimpleBrainDataset(one_case_per_seg, train_transforms)
    bababooey = BrainXLDataset(one_case_per_seg,
                               Compose([ToTensord(keys=["img", "seg"]),
                                        ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0)]),
                               n_pseudo=7)

    loader = DataLoader(bababooey, batch_size=1, shuffle=False, num_workers=0)
    for i,batte in enumerate(loader):
        img = batte["img"].numpy()
        seg = batte["seg"].numpy()

        img = np.moveaxis(img[0], source=0, destination=-1)
        seg = np.moveaxis(np.uint8(seg[0] * 255), source=0, destination=-1)
        print(img.shape)
        print(seg.shape)




    train_datset = MultiModalDataset(tr_cases, train_transforms)
    loader = DataLoader(train_datset, batch_size=2, shuffle=False, num_workers=0)
    for i,batte in enumerate(loader):
        img_50 = batte["img_50"].numpy()
        img_70 = batte["img_70"].numpy()
        img_120 = batte["img_120"].numpy()
        mri = batte["mri"].numpy()
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
        mri = np.moveaxis(mri[0], source=0, destination=-1)
        seg = np.moveaxis(np.uint8(seg[0] * 255), source=0, destination=-1)
        cv2.imshow('50keV', im50[:, :, 1])
        cv2.imshow('70keV', im70[:, :, 1])
        cv2.imshow('120keV', im120[:, :, 1])
        cv2.imshow('MRI', mri)
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
    """


