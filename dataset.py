
import os
import os.path
import cv2
import numpy as np

import torch

from torch.utils.data import Dataset
import torch.utils.data as data
from torchvision import transforms

import util.transform as transform

def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
    else:
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            # maskname = basename + '_seg' + '.png'
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                print('cannot find the mask:', maskpath)

    return img_paths, mask_paths

class ADE20K(Dataset):
    def __init__(self, dataset_path='./dataset/ADEChallengeData2016', split='train', transform=None) :
        self.split = split
        self.images, self.labels = _get_ade20k_pairs(dataset_path, split)
        self.transform = transform
        self.ignore_index = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index],cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img)
        label = cv2.imread(self.labels[index],cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            img, label = self.transform(img, label)
            label = torch.LongTensor(np.array(label).astype('int32') - 1)
        return img, label

def get_dataloader(args):
   
    mean = [0.485, 0.456, 0.406]
    mean_lst = [item * 255 for item in mean]
    std = [0.229, 0.224, 0.225]
    std_lst = [item * 255 for item in std]

    train_transform = transform.Compose([
            transform.RandScale([0.5, 2]),
            transform.RandRotate([-10, 10], padding=mean_lst, ignore_label=0),
            transform.RandomHorizontalFlip(),
            transform.Crop([480, 480], crop_type='rand', padding=mean_lst, ignore_label=0),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
            ])

    test_transform = transform.Compose([
            transform.Crop([480, 480], crop_type='center', padding=mean_lst, ignore_label=0),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
            ])

    train_dataset = ADE20K(split='train', transform = train_transform)
    test_dataset = ADE20K(split='val',transform=test_transform )

    train_loader = data.DataLoader(dataset=train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)

    return train_loader,test_loader
