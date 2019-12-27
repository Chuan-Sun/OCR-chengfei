import codecs
from os.path import join as pjoin
from PIL import Image

import torch
from torchvision.transforms import ToTensor, Grayscale, Resize, Normalize

from config import train_opt
from crnn import CRNN
from ctc_decode import greedy_search


def init_parameters():
    """
    初始化label_dic，它与train.py中label_dic的操作相反
    """
    path_label_list = pjoin(train_opt['data_dir'], 'label_list.txt')

    with codecs.open(path_label_list, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            train_opt['label_dic'][int(parts[1])] = parts[0]
        train_opt['label_dim'] = len(train_opt['label_dic'])


def preprocess(im):
    """
    Grayscale -> Resize -> ToTensor -> Normalize
    :param im:
    :return:
    """
    im = Grayscale()(im)

    # Resize
    im_w, im_h = im.size
    ratio = im_w / im_h
    resized_h = train_opt['input_size'][1]
    resized_w = int(resized_h * ratio)
    im = Resize((resized_h, resized_w))(im)

    im = ToTensor()(im)
    im = Normalize((im.mean(),), (im.std(),))(im)
    return im


def infer_single_image(im_path, model_path):
    init_parameters()

    model: CRNN = CRNN(im_h=train_opt['input_size'][1],
                       n_channel=1,
                       n_class=train_opt['label_dim'] + 1,
                       n_hidden=256)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    im = Image.open(im_path)
    im = preprocess(im)
    im = im.unsqueeze(0)

    with torch.no_grad():
        pred = model(im)
        pred = greedy_search(pred)[0].tolist()
        pred = [train_opt['label_dic'][i] for i in pred]
    return pred


if __name__ == '__main__':
    text = infer_single_image(im_path='../training_data/images_en/000002_0.jpg',
                              model_path='checkpoints/model_epoch_90.pth')
    print(text)
