import numpy as np
from monai.data import GridPatchDataset, DataLoader, PatchIter, PatchDataset, Dataset
from monai.transforms import RandShiftIntensity, RandSpatialCropSamples, Compose, RandFlipd
import SimpleITK as sitk
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union
from torch.utils.data import Subset
import collections.abc
from monai.transforms import Compose, Randomizable, Transform, apply_transform, RandRotated, RandAffined, ToTensord


class Dataset2hD(Dataset):
    def __init__(self, dataurls: Sequence, transform: Optional[Callable] = None) -> None:
        '''
        :param data: Tuples of img,seg files
        :param transform:
        '''

        img = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(dataurls[0][0])),0)
        indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(dataurls[0][1])),0)
        GM = indxs == 2
        WM = indxs == 1
        seg = np.concatenate( (WM,GM),axis=0)
        for k in dataurls[1:]:
            # cimg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(f"{datafolder}/{k}/rawavg.nii")),0)
            # cseg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(f"{datafolder}/{k}/aseg.nii")),0)

            indxs =  np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[1])),0)
            cimg = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(k[0])), 0)
            valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
            cimg = cimg[:,valid_slices[:,0],:,:]
            indxs = indxs[:,valid_slices[:,0],:,:]
            GM = indxs == 2
            WM = indxs == 1


            cseg = np.concatenate((WM, GM), axis=0)

            # img = np.concatenate((img, cimg),axis=1)
            # seg = np.concatenate((seg, cseg), axis=1)
            img = np.concatenate((img, cimg), axis=1)
            seg = np.concatenate((seg, cseg), axis=1)

        # lets just look at AUG for 1 slice
        if 0:
            img120 = img[:,120, :, :]
            seg120 = seg[:,120, :, :]
            for islice in range(img.shape[1]):
                img[:,islice, :, :] = img120
                seg[:,islice, :, :] = seg120

        img[img<0] = 0
        img[img > 100] = 100
        images = {"img": (img-30.0)/5.0, "seg": seg.astype(np.uint8)}
        super().__init__( images, transform )

    def __len__(self) -> int:
        return self.data["img"].shape[1]-2

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        #data_i = self.data[index]
        imgblock = self.data["img"][0,index:index+3,:,:]
        segblock = self.data["seg"][:,index+1,:,:]
        data_i = {"img": imgblock, "seg": segblock}
        #data_i = self.data[index]

        return apply_transform(self.transform, data_i) if self.transform is not None else data_i



    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            #imgblock = np.concatenate( (self.data["img"][index],self.data["img"][index+1],self.data["img"][index+2]))
            #segblock = np.concatenate( (self.data["seg"][index], self.data["seg"][index + 1], self.data["seg"][index + 2]))
            #return {"img":imgblock,"seg":segblock}
            return Subset(dataset=self, indices=index)
        return self._transform(index)


if __name__ == "__main__":
    import cv2

    datafolder = "/home/fi5666wi/Documents/Brain_CT_Project/DL"
    train_transforms = Compose(
        [RandAffined(keys=["img", "seg"], mode=["bilinear", "nearest"], prob=0.9, shear_range=[(0.1), (0.1), (0.1)]),
         ToTensord(keys=["img", "seg"])])
    training_cases = [(f"{datafolder}/Mc43_M70_l_T1.nii", f"{datafolder}/Mc43_seg3.nii"),
                      (f"{datafolder}/Mbr57_M70_l_T1.nii", f"{datafolder}/Mbr57_seg3.nii"),
                      (f"{datafolder}/Ck79_M70_l_T1.nii", f"{datafolder}/Ck79_seg3.nii")]
    train_dataset = Dataset2hD(training_cases, train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    print(len(train_loader))
    for i,batch in enumerate(train_loader):
        print(i)
        img = batch["img"].numpy()

        img = np.moveaxis(img[0], source=0, destination=-1)
        cv2.imshow('CT image',img[:,:,1])
        cv2.waitKey(500)
        if i > 25:
            break


