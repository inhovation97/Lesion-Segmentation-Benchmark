import os
import imageio

import numpy as np
import torch
import torchvision.transforms as transforms
import glob
import cv2

# Save images to folder and create a custom dataloader that loads them from their path. More involved than method 1 but allows for greater flexibility
# Requires 3 functions: __init__ to initialize the object, and __len__ and __get__item for pytorch purposes. More functions can be added as needed, but those 3 are necessary for it to function with pytorch
class myDataSet(object):

    def __init__(self, path_images, path_masks, transforms):
        "Initialization"
        self.all_path_images = sorted(path_images)
        self.all_path_masks = sorted(path_masks)
        self.transforms = transforms

    def __len__(self):
        "Returns length of dataset"
        return len(self.all_path_images)  

    def __getitem__(self, index):
        "Return next item of dataset"
        
        if torch.is_tensor(index):        # 인덱스가 tensor 형태일 수 있으니 리스트 형태로 바꿔준다.
            index = index.tolist()
        
        # Define path to current image and corresponding mask
        path_img = self.all_path_images[index]
        path_mask = self.all_path_masks[index]

        # Load image and mask:
        #     .jpeg has 3 channels, channels recorded last
        #     .jpeg records values as intensities from 0 to 255
        #     masks for some reason have values different to 0 or 255: 0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255
        img_bgr = cv2.imread(path_img) 
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # cv2는 채널이 BGR로 저장된다 -> 출력할 때 RGB로 바꿔줘야함
        img = img / 255  # 픽셀 값들을 0~1로 변환한다
        
        mask = cv2.imread(path_mask)[:, :, 0] / 255  # 마스크의 채널은 1개만 있으면 된다
        mask = mask.round() # binarize to 0 or 1 (이진분류)
        
        # note, resizing happens inside transforms
        
        # convert to Tensors and fix the dimentions
        img = torch.FloatTensor(np.transpose(img, [2, 0 ,1])) # Pytorch uses the channels in the first dimension
        mask = torch.FloatTensor(mask).unsqueeze(0) # Adding channel dimension to label
        
        # apply transforms/augmentation to both image and mask together
        sample = torch.cat((img, mask), 0) # insures that the same transform is applied
        sample = self.transforms(sample)
        img = sample[:img.shape[0], ...]
        mask = sample[img.shape[0]:, ...]

        return img, mask