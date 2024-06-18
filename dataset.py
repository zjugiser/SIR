import os
import cv2
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io.image import read_image
from torchvision.transforms import functional as F, InterpolationMode
from config import Config
from utils.my_resnet50_weights import MyResNet50Weights
from utils.my_vgg16_bn_weights import MyVGG16BNWeights
from utils.my_efficientnetb0_weights import MyEfficientNetB0Weights
import random

if Config.dataset_name == 'WHDLD':
    gray_scale_to_one_hot_dict = {
        128: 0,
        76: 1,
        170: 2,
        226: 3,
        150: 4,
        29: 5
    }
else:
    gray_scale_to_one_hot_dict = {
        255: 0,
        29: 1,
        179: 2,
        150: 3,
        226: 4,
        76: 5
    }


class PILToLongTensor(object):
    def __call__(self, pic):
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.long()
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()
        for key, val in gray_scale_to_one_hot_dict.items():
            img[img == key] = val
        if Config.pretrain_name == 'resnet50':
            img = F.resize(img.unsqueeze(0), 232, interpolation=InterpolationMode.BILINEAR)
            img = F.center_crop(img, 224)
            img = img.reshape(224, 224)
        elif Config.pretrain_name == 'vgg16':
            img = F.resize(img.unsqueeze(0), 224, interpolation=InterpolationMode.BILINEAR)
            img = img.reshape(224, 224)
        else:
            img = F.resize(img.unsqueeze(0), 256, interpolation=InterpolationMode.BILINEAR)
            img = F.center_crop(img, 224)
            img = img.reshape(224, 224)
        return img


class SegDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', num_classes=6, appoint_size=(256, 256), erode=0, aug=False):
        self.imgs_dir = os.path.join(dataset_dir, Config.img_sub_path, mode)
        self.labels_dir = os.path.join(dataset_dir, Config.label_sub_path, mode)
        self.names = os.listdir(self.labels_dir)
        self.num_classes = num_classes
        self.appoint_size = appoint_size
        self.erode = erode
        self.aug = aug

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        label_path = os.path.join(self.labels_dir, name)
        img_path = os.path.join(self.imgs_dir, name[:-3] + 'jpg')
        if self.aug:
            random_down_factor = random.uniform(1, 5)
            new_size = (int(2448 // random_down_factor), int(2048 // random_down_factor))
        if Config.pretrain_name == 'resnet50':
            image = read_image(img_path)
            my_weights = MyResNet50Weights.DEFAULT
            preprocess = my_weights.transforms()
            img_tensor = preprocess(image)
        elif Config.pretrain_name == 'vgg16':
            image = read_image(img_path)
            my_weights = MyVGG16BNWeights.DEFAULT
            preprocess = my_weights.transforms()
            img_tensor = preprocess(image)
        else:
            image = read_image(img_path)
            my_weights = MyEfficientNetB0Weights.DEFAULT
            preprocess = my_weights.transforms()
            img_tensor = preprocess(image)
        label = Image.open(label_path).convert('L')
        if self.aug:
            label = label.resize((new_size[1], new_size[0]), Image.NEAREST)
        label = label.resize((self.appoint_size[1], self.appoint_size[0]), Image.NEAREST)
        if self.erode > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erode, self.erode))
            label_np = cv2.erode(np.array(label), kernel)
            label = Image.fromarray(label_np)
        label_transform = transforms.Compose([PILToLongTensor()])
        label_tensor = label_transform(label)
        return img_tensor, label_tensor
