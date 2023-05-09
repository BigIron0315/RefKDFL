from __future__ import print_function
from asyncio import WriteTransport

import os
import argparse
import socket
import time
import csv
from timeit import default_timer as timer
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from models import model_dict

#from dataset.cifar100 import get_cifar100_dataloaders
from Dataset.cifar10 import get_cifar10_dataloaders

#from helper.util import adjust_learning_rate, accuracy, AverageMeter
from train import accuracy, AverageMeter, validate
from train import train_model
from pthflops import count_ops

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--WeightCopy', type=int, default='0', help='weight copy')
    parser.add_argument('--dataRatio', type=float, default='1.0', help='dataSet ratio')
    parser.add_argument('--iter', type=int, default='1', help='iteration')

    # dataset
    parser.add_argument('--model', type=str, default='Basic_3block_2_64')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()

    # set the path according to the environment
    
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'
    

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name, str(opt.dataRatio))
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def main():
    opt = parse_option()
    print('========== model name : ', opt.model[0:5])
    if (opt.model[0:5] == 'Bottl'):
        dir_path = 'CSV/Bottleneck'
        print('========== Bottleneck ')
    elif (opt.model[0:5] == 'Resid'):
        dir_path = 'CSV/Residual'
        print('========== Residual ')
    elif (opt.model[0:5] == 'Basic'):
        dir_path = 'CSV/Basic'
        print('========== Basic ')
    
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    file_name = opt.model + '_' + opt.dataset +'_' + str(opt.learning_rate) + '_iter_' + str(opt.iter) + '.csv'
    with open('{file_path}'.format(file_path=os.path.join(dir_path, file_name)), 'w') as csvfile:
        fieldnames = ['epoch', 'test_acc', 'time_per_epoch', 'learningRate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        best_acc = 0

        # dataloader
        if opt.dataset == 'cifar10':
            train_loader, val_loader = get_cifar10_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, dataRatio = opt.dataRatio)
            n_cls = 10
        else:
            raise NotImplementedError(opt.dataset)

        # model
        model = model_dict[opt.model](num_classes=n_cls)
        print_size_of_model(model)
        print('model : ', model)


        # optimizer
        optimizer = optim.SGD(model.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)

        criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True

        device = 'cuda:0'
        inp = torch.rand(1,3,32,32).to(device)
        count_ops(model, inp, mode='fx')

        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

        for epoch in range(1, opt.epochs + 1):

            adjust_learning_rate(epoch, opt, optimizer)
            print("==> training...learning rate : ", opt.learning_rate)
            start_time = timer()
            train_acc, train_loss = train_model(epoch, train_loader, model, criterion, optimizer, opt)

            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)

            test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

            logger.log_value('test_acc', test_acc, epoch)
            logger.log_value('test_acc_top5', test_acc_top5, epoch)
            logger.log_value('test_loss', test_loss, epoch)
            end_time = timer()
            print('end_time : ', end_time)
            print('duration : ', end_time - start_time)            
            writer.writerow({'epoch': epoch, 'test_acc': test_acc, 'time_per_epoch': end_time - start_time, 'learningRate' : opt.learning_rate})

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
                print('saving the best model!')
                torch.save(state, save_file)

            if epoch % opt.save_freq == 0:
                print('==> Saving...')
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'accuracy': test_acc,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(state, save_file)

        print('best accuracy:', best_acc)

        state = {
            'opt': opt,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
        torch.save(state, save_file)


if __name__ == '__main__':
    main()
