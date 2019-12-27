import codecs
import random
import numpy as np
import logging
import os
from os.path import join as pjoin

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# my file
from config import train_opt
from crnn import CRNN
from dataset import TrainSet, ValidateSet
from ctc_decode import greedy_search


# TODO: simplify
def init_log_config():
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = 'logs'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = pjoin(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)


def init_parameters():
    """ 初始化label_dic """
    path_label_list = pjoin(train_opt['data_dir'], 'label_list.txt')

    with codecs.open(path_label_list, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            train_opt['label_dic'][parts[0]] = int(parts[1])
        train_opt['label_dim'] = len(train_opt['label_dic'])


def checkpoint():
    os.makedirs('checkpoints', exist_ok=True)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), "checkpoints/model_epoch_{}.pth".format(epoch))


def load_pre_trained_params():
    if train_opt['model'] != '':
        model.load_state_dict(torch.load(train_opt['model']))
        logger.info('load param from pre-trained model')


def train():
    model.train()
    epoch_loss = 0
    for iteration, batch in enumerate(train_loader, 1):
        # Input
        im, label, label_lengths = batch[0].cuda(0), batch[1].cuda(0), batch[2]

        # FP
        pred = model(im)
        pred_lengths = torch.tensor([pred.shape[0]] * pred.shape[1], dtype=torch.int32)

        # Loss
        loss = criterion(pred, label, pred_lengths, label_lengths)
        epoch_loss += loss.item()

        # BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            logger.info("Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(train_loader), loss.item()))

    logger.info("Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_loader)))


def validate():
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for batch in validate_loader:
            im, label, label_lengths = batch[0].cuda(0), batch[1].numpy(), batch[2]
            pred = model(im)
            pred = greedy_search(pred, blank=0)

            # Count correct
            for i in range(label_lengths.shape[0]):
                if pred[i].tolist() == label[i, :label_lengths[i]].tolist():
                    n_correct += 1

    accuracy = n_correct / len(validate_loader.dataset)
    logger.info("Validating set accuracy: {}/{} ({:.0f}%)".format(n_correct, len(validate_loader.dataset), 100. * accuracy))
    return accuracy


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    cudnn.benchmark = True

    logger = None
    init_log_config()
    init_parameters()

    # Note: Win系统多进程情况下，train_opt跨文件全局变量会被重新初始化，导致label_dic被清空，
    # 所以此处将label_dic单独传至类变量中。Linux系统下好像没这事儿。
    train_set    = TrainSet('../training_data', 'train.txt', label_dic=train_opt['label_dic'])
    validate_set = ValidateSet('../training_data', 'train.txt', label_dic=train_opt['label_dic'])
    train_loader    = DataLoader(train_set,    batch_size=train_opt['batch_size'], shuffle=True,  num_workers=6)
    validate_loader = DataLoader(validate_set, batch_size=train_opt['batch_size'], shuffle=False, num_workers=6)

    # Creat model and load pre-trained parameters
    model: CRNN = CRNN(im_h=train_opt['input_size'][1],
                       n_channel=1,
                       n_class=train_opt['label_dim'] + 1,
                       n_hidden=256)
    model.cuda(0)
    load_pre_trained_params()

    criterion = CTCLoss()

    optimizer = Adam(model.parameters(), train_opt['lr'], weight_decay=1e-3)

    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    for epoch in range(1, train_opt['n_epochs']+1):
        train()
        acc = validate()
        scheduler.step(epoch)
        checkpoint()
