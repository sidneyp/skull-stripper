import os
import numpy as np

import torch
import nibabel as nib
from glob import glob
import scipy.io
import random
from PIL import Image
import elastic_transform as elt
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

def load_nifty(full_file_name):
    img = nib.load(full_file_name)
    #dtype = img.get_data_dtype()  # F8 is 64-bit floating-point Number
    data = img.get_fdata()
    return data

def tensor_2_numpy_image(tensor):
    img_out = np.moveaxis(tensor.numpy()[:,:,:].squeeze(), 0, -1)
    return img_out

def to_img(batch_of_images):
    img = batch_of_images[0]
    img = tensor_2_numpy_image(img)
    img-=np.min(img[:])
    img *= 255.0/img.max()
    img = img.astype(np.uint8)
    return img

def cvt1to3channels(one_channel):
    return np.stack((one_channel,)*3, axis=-1)

def load_dataset(src_path, mask_path, validation_portion=0.05):
    # 1- set the paths
    src_format = 'mat'
    mask_format = 'nii'
    src_file_format = '*.{}'.format(src_format)
    mask_file_format = '*.{}'.format(mask_format)
    all_src_img = glob(os.path.join(src_path,src_file_format))
    all_mask_img = glob(os.path.join(mask_path,mask_file_format))
    all_src_img.sort()
    all_mask_img.sort()

    # 2- Find the matching pairs
    src_msk_file_pair_list = []
    for i,src_f in enumerate(all_src_img):
        base_src_name = os.path.basename(src_f)
        base_src_name = base_src_name.split('.')[0]
        src_id1, src_id2 = base_src_name.split('_')[1:3]
        for j,msk_f in enumerate(all_mask_img):
            base_msk_name = os.path.basename(msk_f)
            base_msk_name = base_msk_name.split('.')[0]
            msk_id1, msk_id2 = base_msk_name.split('_')[1:3]
            if src_id1 == msk_id1 and src_id2 == msk_id2:
                src_msk_file_pair_list.append([src_f,msk_f])

    # 3- load every single frame and stores it into a list
    src = []
    msk = []
    for i in range(len(src_msk_file_pair_list)):
        src_f, msk_f = src_msk_file_pair_list[i]
        mat = scipy.io.loadmat(src_f)
        if 'N' in mat:
            src_mat = mat["N"]
        msk_mat = load_nifty(msk_f)
        for j in range(min(src_mat.shape[2], msk_mat.shape[2])):
            src.append(np.uint8(src_mat[:,:,j]))
            msk.append(np.uint8(msk_mat[:,:,j]))

    src = np.array(src)
    msk = np.array(msk)

    validation_size = int(len(src) * validation_portion)
    train_size = len(src)-validation_size

    src_train, src_val = np.split(src, [train_size])
    msk_train, msk_val = np.split(msk, [train_size])

    return src_train, msk_train, src_val, msk_val


# 5- Define Dataset model
class MouseMRIDS(Dataset):
    def __init__(self,
                 src,
                 msk,
                 transform = None,
                 augmentation=True):

        self.src = src
        self.msk = msk

        indices_with_problem = np.where(np.logical_and(
                np.min(self.msk, axis=(1,2)) == 0,
                np.max(self.msk, axis=(1,2)) == 1) == False)

        if len(indices_with_problem) > 0:
            for i in indices_with_problem:
                self.src = np.delete(self.src, i, axis=0)
                self.msk = np.delete(self.msk, i, axis=0)

        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.src)

    def __getitem__(self,idx):
        if random.random() > 0.5 or not self.augmentation:
            src_img = self.src[idx]
            msk_img = self.msk[idx]
        else:
            src_img, msk_img = elt.get_elastic_transforms(self.src[idx],
                                                      self.msk[idx])

        src_im = Image.fromarray(np.uint8(cvt1to3channels(src_img)))
        msk_im = Image.fromarray(np.uint8(cvt1to3channels(msk_img)))

        # Apply the same trasnformation to the two images
        if self.transform:
            if random.random() > 0.5 and self.augmentation:
               src_im = F.vflip(src_im)
               msk_im = F.vflip(msk_im)
            if random.random() > 0.5 and self.augmentation:
               src_im = F.hflip(src_im)
               msk_im = F.hflip(msk_im)
            if random.random() > 0.5 and self.augmentation:
               angle=np.random.choice([90,180,270])
               src_im = F.rotate(src_im,angle)
               msk_im = F.rotate(msk_im,angle)

            src_im = self.transform(src_im)
            msk_im = self.transform(msk_im)

            msk_im = (msk_im - torch.min(msk_im)) / torch.max(msk_im)

        return src_im,\
                msk_im[1,:,:].expand(1,-1,-1).type(torch.float)

