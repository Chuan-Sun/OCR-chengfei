import os
from os.path import join as pjoin
import codecs
from PIL import Image

from config import train_opt


def quantile_width(path_images):
    """
    图片宽度的二十分位数
    :param path_images:
    :return:
    """
    im_paths = [pjoin(path_images, file) for file in os.listdir(path_images)]
    ratios = []
    for p in im_paths:
        im = Image.open(p)
        w, h = im.size
        ratios.append(w / h)

    quantile_ratio = sorted(ratios, reverse=True)[len(ratios)//20]
    print(f"The 20 quantile ratio is {quantile_ratio}")
    print(f"The 20 quantile width is {quantile_ratio*train_opt['input_size'][1]}")


def quantile_label_length(im_file_path):
    with codecs.open(im_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
        label_lengths = []
        for line in lines:
            name, label = line.split(maxsplit=1)
            label_lengths.append(len(label))

    quantile_length = sorted(label_lengths, reverse=True)[len(label_lengths)//20]
    print(f"The 20 quantile label length is {quantile_length}")


if __name__ == '__main__':
    quantile_width('../training_data/images_en')
    quantile_label_length('../training_data/train_en.txt')
