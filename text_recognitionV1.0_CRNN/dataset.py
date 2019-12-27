import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ColorJitter, Grayscale, Pad, Normalize, RandomRotation, Resize

import os
from os.path import join as pjoin
import numpy as np
from PIL import Image
import random

from config import train_opt


# TODO: Random use data augmentation, not always
def preprocess(im):
    """
    1.Jitter color -> 2.Grayscale -> 3.Mean&Std -> 4.Expand -> 5.Rotate -> 6.Resize -> 7.ToTensor -> 8.Normalize
    :param im:
    :return:
    """
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im_w, im_h = im.size

    # 1.Data augmentation: Jitter color
    im = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(im)

    # 2.-> Grayscale
    im = Grayscale()(im)

    # 3.Calculate the mean and std of the image for Normalizing later
    mean = np.mean(im).round().astype('uint8')
    std  = np.std(im)

    # 4.Data augmentation: expand
    random_ratio = random.uniform(0, 0.6)
    n_pad = int(im_h * random_ratio / 2)
    im = Pad(n_pad, fill=(mean,))(im)

    # 5.Data augmentation: rotate
    im = RandomRotation(5, resample=Image.BILINEAR, expand=False)(im)

    # 6.Resize
    im_w, im_h = im.size
    ratio = min(float(train_opt['input_size'][1])/im_h, float(train_opt['input_size'][2]/im_w))
    resized_h = int(round(im_h*ratio))
    resized_w = int(round(im_w*ratio))
    im = Resize((resized_h, resized_w))(im)
    h_off = (train_opt['input_size'][1] - resized_h) / 2
    w_off = (train_opt['input_size'][2] - resized_w) / 2
    bg = mean * np.ones((train_opt['input_size'][1], train_opt['input_size'][2]), np.uint8)
    bg = Image.fromarray(bg)
    bg.paste(im, (np.random.randint(0, w_off+1), int(h_off)))

    # 7.-> Tensor
    im = ToTensor()(bg)
    # 8.Normalize
    im = (im - mean/255) / (std/255)

    return im


class _DataSet(Dataset):

    def __init__(self, data_dir, file_name, label_dic):
        super(Dataset, self).__init__()

        # load all lines in "file_name"
        file_path = pjoin(data_dir, file_name)
        self.sample_list = [line.strip().split(maxsplit=1) for line in open(file_path, encoding='utf-8')]

        self.label_dic = label_dic

    def __getitem__(self, index):
        parts = self.sample_list[index]        # path + label

        # image
        im_path = parts[0]
        im = Image.open(im_path)
        im = preprocess(im)

        # label
        label = parts[1]
        label = [int(self.label_dic[c]) for c in label]
        label_lengths = torch.tensor(len(label), dtype=torch.int32)
        label += [0] * (train_opt['max_label_length'] - len(label))
        label = torch.tensor(label, dtype=torch.int32)

        return im, label, label_lengths

    def __len__(self):
        return len(self.sample_list)


class TrainSet(_DataSet):
    def __init__(self, data_dir, train_file, label_dic):
        super(TrainSet, self).__init__(data_dir, train_file, label_dic)


class ValidateSet(_DataSet):
    def __init__(self, data_dir, validate_file, label_dic):
        super(ValidateSet, self).__init__(data_dir, validate_file, label_dic)
        self.sample_list = self.sample_list[:100]


if __name__ == '__main__':
    from train import init_parameters
    init_parameters()
    train_dataset = TrainSet(data_dir='../training_data', train_file='train.txt')
    for _ in train_dataset:
        pass
