"""
源.list文件格式：
[ w, h, file_name, label ]

预处理后的数据集格式为：
File_name\tLabel
例如：1_1.jpg	xybLoveCharlotte

output:
训练用的 train.txt
记录所有支持字符的 label_list.txt
"""
from langconv import *
import codecs
import os
import random
from os.path import join as pjoin

from config import train_opt


def read_ims_list(path_ims_list):
    """
    读取 train.list 文件
    """
    ims_info_dic = {}
    with open(path_ims_list, 'r', encoding='utf-8') as f:
        for line in f:
            # Chinese dataset
            # parts = line.strip().split(maxsplit=3)
            # w, h, file, label = parts[0], parts[1], parts[2], parts[3]
            # ims_info_dic[file] = {'label': label, 'w': int(w)}

            # English dataset
            parts = line.strip().split(maxsplit=1)
            file, label = parts[0], parts[1]
            ims_info_dic[file] = {'label': label}
    return ims_info_dic


def modify_ch(label):
    # 繁体 -> 简体
    label = Converter("zh-hans").convert(label)

    # 大写 -> 小写
    label = label.lower()

    # 删除空格
    label = label.replace(' ', '')

    # 删除符号
    for ch in label:
        if (not '\u4e00' <= ch <= '\u9fff') and (not ch.isalnum()):
            label = label.replace(ch, '')

    return label


def pipeline(dataset_dir):
    path_ims        = pjoin(dataset_dir, "images_en")
    path_ims_list   = pjoin(dataset_dir, "train_en.txt")
    path_train_list = pjoin(dataset_dir, "train.txt")
    path_label_list = pjoin(dataset_dir, "label_list.txt")

    # 读取数据信息
    file_info_dic = read_ims_list(path_ims_list)

    # 创建 train.txt
    label_set = set()
    with codecs.open(path_train_list, 'w', encoding='utf-8') as f:
        for file, info in file_info_dic.items():
            label = info['label']
            label = modify_ch(label)

            # 异常: 标签为空 or 标签过长
            if label == '' or len(label) > train_opt['max_label_length']:
                continue

            for e in label:
                label_set.add(e)

            f.write("{0}\t{1}\n".format(pjoin(path_ims, file), label))

    # 创建 label_list.txt
    lable_list = list(label_set)
    lable_list.sort()
    print("class num: {0}".format(len(lable_list)))
    with codecs.open(path_label_list, "w", encoding='utf-8') as f:
        for id, c in enumerate(lable_list):
            f.write("{0}\t{1}\n".format(c, id+1))


if __name__ == '__main__':
    random.seed(0)
    pipeline(dataset_dir="../training_data")
