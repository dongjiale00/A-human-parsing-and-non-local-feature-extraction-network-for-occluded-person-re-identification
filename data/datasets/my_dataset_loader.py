# encoding: utf-8


import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import math
import random
import torch
from torchvision.transforms import Resize
from config import cfg


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, align_target_path = self.dataset[index]
        img = read_image(img_path)
        if osp.exists(align_target_path):
            align_target=Image.open(align_target_path).convert('L')
            align_target=np.array(align_target).astype('int32')
        
        else:
            H=64;W=32
            align_target=np.zeros((H,W)).astype('int32')
            h_margin=int((H)/(6))
            for h in range(H):
                for w in range(W):
                    align_target[h,w]=int(h/h_margin)+1 if int(h/h_margin)+1 < 7 else 6
        
        
        if self.transform is not None:
            img = self.transform(img)
        '''
        _, H, W = img.size()
        
        H_aware = np.tile(np.linspace(-1.0, 1.0, H).astype('float32').reshape(1,-1).T,(1,W)).reshape(1,H,W)
        W_aware = np.tile(np.linspace(-1.0, 1.0, W).astype('float32').reshape(1,-1),(H,1)).reshape(1,H,W)
        
        img=np.concatenate((img, H_aware, W_aware))
        '''
        return img, pid, camid, img_path, align_target, align_target_path

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img, label):

        if random.uniform(0, 1) >= self.probability:
            return img, label
        
        mean = torch.mean(img, (1, 2))
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = mean[0]
                label[x1:x1 + h, y1:y1 + w] = int(0)
                return img, label

        return img, label
        
class HorizontalFlip(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img, label):

        if random.uniform(0, 1) >= self.probability:
            return img, label
        
        img = torch.from_numpy(img.numpy()[:,:,::-1].copy())
        label = label[:,::-1]
        
        return img, label

class ImageDataset_train(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.random_earse= RandomErasing(probability=0.5)
        self.flip = HorizontalFlip(probability=0.5)
        self.img_pil_crop = Image.new('RGB', (256, 256), (0, 0, 0))
        self.idx = [i for i in range(len(self.dataset))]

    def __len__(self):
        return len(self.dataset)

    def get_params(self, img0, scale=(0.01, 0.03), ratio=(0.3, 3.3)): #随机my2
        img = img0.copy()
        img_array = np.array(img).transpose(2, 0, 1)  # transpose (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_array)
        img_c, img_h, img_w = img_tensor.shape
        area = img_h * img_w

        for _ in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                # 只获取4个角
                i = random.sample([0,img_h-h], 1)[0]
                j = random.sample([0,img_w-w], 1)[0]

        return i, j, h, w

    def occluded(self, img_pil0,img_pil_crop, position, size0, align_target_pil):
        img_pil=img_pil0.copy()
        resize = Resize(
        size=(size0[0], size0[1]),  # (height, width)
        )
        img_pil_crop=resize(img_pil_crop)
        img_pil.paste(img_pil_crop,(position[1],position[0],position[1] + size0[1],position[0] + size0[0]))
        align_target_pil
        return img_pil

    def __getitem__(self, index):
        img_path, pid, camid, align_target_path = self.dataset[index]
        img = read_image(img_path)
        img_pil = img.copy()

        if osp.exists(align_target_path):
            align_target=Image.open(align_target_path).convert('L').resize((32,64),Image.NEAREST)
            align_target=np.array(align_target).astype('int32')
        
        else:
            H=64;W=32
            align_target=np.zeros((H,W)).astype('int32')
            h_margin=int((H)/(5))
            for h in range(H):
                for w in range(W):
                    align_target[h,w]=int(h/h_margin)+1 if int(h/h_margin)+1 < 6 else 5
        align_target_pil = align_target.copy()

        # 随机一张图片作为贴图
        img_pil_idx = Image.open(self.dataset[random.choice(self.idx)][1])
        img_pil_idx = resize(img_pil_idx)
        scale = (0.02, 0.2)
        ratio = (0.3, 3.3)
        i, j, h, w = self.get_params(img_pil_idx, scale, ratio)
        img_pil_crop = img_pil_idx.crop((j, i, j + w, i + h))  # (l,u,r,d)

        size = random.sample([8192, 16384], 1)[0]
        rl = size // 256
        position_l = [0, 0]
        size_l = [256, rl]
        #
        rr = size // 256
        position_r = [0, 128 - rr]
        size_r = [256, rr]
        #
        ru = size // 128
        position_u = [0, 0]
        size_u = [ru, 128]
        #
        rd = size // 128
        position_d = [256 - rd, 0]
        size_d = [rd, 128]

        # 生成图片
        img_u = self.occluded(img_pil, img_pil_crop, position_u, size_u)
        img_l = self.occluded(img_pil, img_pil_crop, position_l, size_l)
        img_r = self.occluded(img_pil, img_pil_crop, position_r, size_r)
        img_d = self.occluded(img_pil, img_pil_crop, position_d, size_d)

        # 变换
        img = self.add_transform(img).unsqueeze(0)
        img_l = self.add_transform(img_l).unsqueeze(0)
        img_r = self.add_transform(img_r).unsqueeze(0)
        img_u = self.transform(img_u).unsqueeze(0)
        img_d = self.transform(img_d).unsqueeze(0)


        if self.transform is not None:
            img = self.transform(img)
        
        #Filp 0.5
        img, align_target = self.flip(img, align_target)
        #Random Earse 0.5
        img, align_target = self.random_earse(img, align_target)
        
        '''
        _, H, W = img.size()
        
        H_aware = np.tile(np.linspace(-1.0, 1.0, H).astype('float32').reshape(1,-1).T,(1,W)).reshape(1,H,W)
        W_aware = np.tile(np.linspace(-1.0, 1.0, W).astype('float32').reshape(1,-1),(H,1)).reshape(1,H,W)
        
        img=np.concatenate((img, H_aware, W_aware))
        '''
        
        return img, pid, camid, img_path, align_target, align_target_path
