# encoding: utf-8


import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import math
import random
import torch
import os
import torchvision.transforms as T

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


class RandomErasing_Background(object):
    def __init__(self, EPSILON=0.5, root=None):
        self.EPSILON = EPSILON
        self.root = root
        self.occ_imgs = os.listdir(self.root)

        for img in self.occ_imgs:
            if not img.endswith('.jpg'):
                self.occ_imgs.remove(img)

        self.len = len(self.occ_imgs)

    def __call__(self, img, label):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        index = random.randint(0, self.len - 1)

        occ_img = self.occ_imgs[index]
        occ_img = Image.open(os.path.join(self.root, occ_img)).convert('RGB')
        #print(type(img))
        h, w = img.size()[1], img.size()[2]
        h_, w_ = occ_img.height, occ_img.width

        ratio = h_ / w_
        if ratio > 2:
            # re_size = (random.randint(h//2, h), random.randint(w//4, w//2))
            re_size = (h, random.randint(w // 4, w // 2))
            function = T.Compose([
                T.Resize(re_size, interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ])
            occ_img = function(occ_img)
        else:
            # re_size = (random.randint(h//4, h//2), random.randint(w//2, w))
            re_size = (random.randint(h // 4, h // 2), w)
            function = T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                T.Resize(re_size, interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ])
            occ_img = function(occ_img)

        h_, w_ = re_size[0], re_size[1]
        index_ = random.randint(0, 3)
        # points = [(0, 0), (0, w), (h, 0), (h, w)]

        if index_ == 0:
            img[:, 0:h_, 0:w_] = occ_img
            label[ 0:h_, 0:w_] = int(0)
        elif index_ == 1:
            img[:, 0:h_, w - w_:w] = occ_img
            label[ 0:h_, w - w_:w] = int(0)
        elif index_ == 2:
            img[:, h - h_:h, 0:w_] = occ_img
            label[ h - h_:h, 0:w_] = int(0)
        else:
            img[:, h - h_:h, w - w_:w] = occ_img
            label[ h - h_:h, w - w_:w] = int(0)

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
        self.random_earse_background = RandomErasing_Background(EPSILON=1, root='/home/fzl/Projects/FED-Occluded-ReID-main/FED-Occluded-ReID-main/crop_backgrounds/crop_backgrounds')
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, align_target_path = self.dataset[index]
        img = read_image(img_path)
        if osp.exists(align_target_path):
            align_target=Image.open(align_target_path).convert('L').resize((32,64),Image.NEAREST)
            align_target=np.array(align_target).astype('int32')
            # print("you")
        
        else:
            H=64;W=32
            align_target=np.zeros((H,W)).astype('int32')
            h_margin=int((H)/(5))
            # print("meiyou")
            for h in range(H):
                for w in range(W):
                    align_target[h,w]=int(h/h_margin)+1 if int(h/h_margin)+1 < 6 else 5

        img_2 = img.copy()
        align_target_2 = align_target.copy()

        if self.transform is not None:
            img = self.transform(img)
            img_2 = self.transform(img_2)
        #print(img_2)
        img_2, align_target_2 = self.random_earse_background(img_2, align_target_2)

        #Filp 0.5
        img, align_target = self.flip(img, align_target)
        img_2, align_target_2 = self.flip(img, align_target)
        #Random Earse 0.5
        #img, align_target = self.random_earse(img, align_target)
        '''
        _, H, W = img.size()
        
        H_aware = np.tile(np.linspace(-1.0, 1.0, H).astype('float32').reshape(1,-1).T,(1,W)).reshape(1,H,W)
        W_aware = np.tile(np.linspace(-1.0, 1.0, W).astype('float32').reshape(1,-1),(H,1)).reshape(1,H,W)
        
        img=np.concatenate((img, H_aware, W_aware))
        '''


        return img, pid, camid, img_path, align_target, align_target_path, img_2, align_target_2
