from FL_general import *
from FL_methods import *
from utils_resnet import resnet_compound_cifar100, resnet_width_cifar100, resnet_1_8_cifar100, resnet_scale_cifar100, resnet_scale_base_cifar100, resnet8_cifar100, resnet14_cifar100, resnet20_cifar100, resnet26_cifar100, resnet56_cifar100
from utils_resnet_bottleneck import resnet_B_1_8_cifar100, resnet_B_scale_cifar100, resnet_B_scale_base_cifar100, resnet11_B_cifar100, resnet20_B_cifar100, resnet29_B_cifar100, resnet38_B_cifar100, resnet83_B_cifar100, resnet_B_width_cifar100, resnet_B_compound_cifar100
from utils_vgg import vgg_scale, vgg_scale_cifar100, vgg13, vgg13_cifar100, vgg13_compound_cifar100, vgg13_1_8_cifar100
###############################################################################################################################
import os
import argparse
import socket
import time
import torch
import torch.nn as nn
import torch.quantization
from pthflops import count_ops
from torch.utils.tensorboard import SummaryWriter
from Teacher_model.__init__ import model_dict1

# Create a network and a corresponding input

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.99, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--isDistill', type=int, default=1, help='use or not distillation')
    parser.add_argument('--publicRatio', type=float, default=0.1, help='public data ratio')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar100', 'cifar10', 'mnist'], help='dataset')
    parser.add_argument('--Dir', type=float, default=0.3, help='dataset iid')

    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--TA_train', type=int, default=0, help='Train TA model')
    parser.add_argument('--student_train', type=int, default=0, help='student TA model')
    parser.add_argument('--model_name', type=str, default='resnet', choices=['resnet', 'vgg', 'resnet_B'], help='resnet')
    parser.add_argument('--target_ratio', type=float, default = 1.0, help='target ratio')
    parser.add_argument('--pretrained_student', type=int, default=0, help='Temp')
    parser.add_argument('--temperature', type=float, default=16, help='Temp')
    parser.add_argument('--same', type=int, default=0, help='from same network')
    parser.add_argument('--dynTemp', type=int, default=0, help='dynTemp')
    parser.add_argument('--batch', type=int, default=50, help='batch')
    parser.add_argument('--fed_a', type=int, default=0, help='fedAvg')
    parser.add_argument('--fed_b', type=int, default=0, help='Scaffold')
    parser.add_argument('--fed_c', type=int, default=0, help='FedProx')
    parser.add_argument('--fed_d', type=int, default=0, help='RefKDFL')
    parser.add_argument('--dynRatio', type=int, default=1, help='dynRatio')
    parser.add_argument('--arch', type=int, default=1, help='arch')
    opt = parser.parse_args()

    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    return model_path.split('/')[-3]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model : ', model_path)
    model_t = get_teacher_name(model_path)
    model = model_dict1[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    model.cuda()
    print('==> done')
    return model

opt = parse_option()

# Generate IID or Dirichlet distribution
# IID
n_client = 100
if opt.Dir == 0.0:
    data_obj = DatasetObject(dataset='CIFAR100', n_client=n_client, rule='iid', public = opt.publicRatio, unbalanced_sgm=0)
else:
    data_obj = DatasetObject(dataset='CIFAR100', n_client=n_client, rule='Dirichlet', public = opt.publicRatio, rule_arg=opt.Dir)

###

com_amount         = 300
save_period        = 200
act_prob           = 0.1
epoch              = 5
print_per          = 5
weight_decay       = opt.weight_decay
batch_size         = opt.batch
lr_decay_per_round = opt.lr_decay_rate
learning_rate      = opt.learning_rate

num_cls = 100
print('opt Dir {} arch {} num_cls {} target ratio {} cuda {} '.format(opt.Dir, opt.arch, num_cls, opt.target_ratio, torch.cuda.is_available()))

# Model function
if opt.model_name == 'resnet':
    if opt.arch == 1:
        model_KDFL = resnet_scale_cifar100
    else:
        model_KDFL = resnet_scale_base_cifar100
        
    if opt.target_ratio <= 0.125:
        if opt.arch == 4:
            model_func = resnet_B_1_8_cifar100 
        elif opt.arch == 2:
            model_func = resnet_width_cifar100
        elif opt.arch == 3:
            model_func = resnet_compound_cifar100
        else:
            model_func = resnet8_cifar100
    elif opt.target_ratio <= 0.25:
        model_func = resnet14_cifar100
    elif opt.target_ratio <= 0.35:
        model_func = resnet20_cifar100
    elif opt.target_ratio <= 0.5:
        model_func = resnet26_cifar100                                                                                                                                                                                          
    elif opt.target_ratio <= 1:
        model_func = resnet56_cifar100
elif opt.model_name == 'vgg':
    model_KDFL = vgg_scale_cifar100
    if opt.target_ratio <= 0.125:
        if opt.arch == 3:
            model_func = vgg13_compound_cifar100
        else:
            model_func = vgg13_1_8_cifar100
    else:
        model_func = vgg13_cifar100
    print('model is vgg')
elif opt.model_name == 'resnet_B':
    if opt.arch == 1:
        model_KDFL = resnet_B_scale_cifar100
    else:
        model_KDFL = resnet_B_scale_base_cifar100
        
    if opt.target_ratio <= 0.125:
        if opt.arch == 4:
            model_func = resnet_B_1_8_cifar100
        elif opt.arch == 2:
            model_func = resnet_B_width_cifar100
        elif opt.arch == 3:
            model_func = resnet_B_compound_cifar100
        else:
            model_func = resnet11_B_cifar100
    elif opt.target_ratio <= 0.25:
        model_func = resnet20_B_cifar100
    elif opt.target_ratio <= 0.35:
        model_func = resnet29_B_cifar100
    elif opt.target_ratio <= 0.5:
        model_func = resnet38_B_cifar100                                                                                                                                                                                          
    elif opt.target_ratio <= 1:
        model_func = resnet83_B_cifar100

init_model = model_func()
init_model_KD = model_KDFL(opt.target_ratio)
device = 'cuda:0'
model = model_func().to(device)
model_KD = model_KDFL(opt.target_ratio).to(device)

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

if opt.TA_train == 1:
    if opt.model_name == 'resnet':
        ta = model_dict1['TA_Resnet110_CIFAR100'](num_classes=num_cls)
        ta = ta.cuda()
        ta_learning_rate = 0.05
        ta_batch_size = 10
        TA_model = train_TA(data_obj, 'CIFAR100', ta_learning_rate, ta_batch_size, 240, weight_decay, ta, lr_decay_per_round, rand_seed=0, opt=opt)
        model_path = './save/models/TA_Resnet110_CIFAR100/PublicRatio_%d' %(int(opt.publicRatio*100))
        #TA_model = train_TA(data_obj, 'CIFAR100', ta_learning_rate, ta_batch_size, 1, weight_decay, ta, lr_decay_per_round, rand_seed=0, opt=opt)
        

        opt.save_folder = os.path.join(model_path)
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)
        #print(torch.cuda.device_count())
        print('==> Saving...')
        state = {
            'model': TA_model.state_dict(),
        }
        
        save_file = os.path.join(opt.save_folder, 'TA_Resnet110_CIFAR100.pth')
        
        torch.save(state, save_file)
    elif opt.model_name == 'vgg':
        ta = model_dict1['TA_Vgg_CIFAR100'](num_classes=num_cls)
        ta = ta.cuda()
        ta_learning_rate = 0.05
        ta_batch_size = 10
        TA_model = train_TA(data_obj, 'CIFAR100', ta_learning_rate, ta_batch_size, 240, weight_decay, ta, lr_decay_per_round, rand_seed=0, opt=opt)
        model_path = './save/models/TA_Vgg_CIFAR100/PublicRatio_%d' %(int(opt.publicRatio*100))

        opt.save_folder = os.path.join(model_path)
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)
        #print(torch.cuda.device_count())
        print('==> Saving...')
        state = {
            'model': TA_model.state_dict(),
        }
        
        save_file = os.path.join(opt.save_folder, 'TA_Vgg_CIFAR100.pth')
        
        torch.save(state, save_file)
else:
    TA_model = load_teacher(opt.path_t, num_cls)


module_list = nn.ModuleList([])

if not os.path.exists('Output'):
    os.mkdir('Output')

if not os.path.exists('Output/%s' %(data_obj.name)):
    os.mkdir('Output/%s' %(data_obj.name))

# FL Methods

if (opt.fed_a == 1):
    print('FedAvg')
    init_model.cuda()

    [fed_mdls_sel_FedAvg, trn_perf_sel_FedAvg, tst_perf_sel_FedAvg,
    fed_mdls_all_FedAvg, trn_perf_all_FedAvg,
    tst_perf_all_FedAvg] = train_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        lr_decay_per_round=lr_decay_per_round, opt=opt)

if (opt.fed_b == 1):
    print('FedProx')

    mu = 1e-4

    [fed_mdls_sel_FedProx, trn_perf_sel_FedProx, tst_perf_sel_FedProx,
    fed_mdls_all_FedProx, trn_perf_all_FedProx,
    tst_perf_all_FedProx] = train_FedProx(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                        model_func=model_func, init_model=init_model, save_period=save_period,
                                        mu=mu, lr_decay_per_round=lr_decay_per_round, opt=opt)

if (opt.fed_c == 1):
    init_model.cuda()
    print('Scaffold')
    n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / n_client
    n_iter_per_epoch  = np.ceil(n_data_per_client/batch_size)
    n_minibatch = (epoch*n_iter_per_epoch).astype(np.int64)
    print_per_ = print_per*n_iter_per_epoch

    [fed_mdls_sel_SCAFFOLD, trn_perf_sel_SCAFFOLD, tst_perf_sel_SCAFFOLD,
    fed_mdls_all_SCAFFOLD, trn_perf_all_SCAFFOLD,
    tst_perf_all_SCAFFOLD] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                                            batch_size=batch_size, n_minibatch=n_minibatch, com_amount=com_amount,
                                            print_per=print_per_, weight_decay=weight_decay, model_func=model_func,
                                            init_model=init_model, save_period=save_period, lr_decay_per_round=lr_decay_per_round, opt=opt)

if (opt.fed_d == 1):
    print('RefKDFL')
    init_model_KD.cuda()

    [fed_mdls_sel_FedAvg_KDFL, trn_perf_sel_FedAvg_KDFL, tst_perf_sel_FedAvg_KDFL,
    fed_mdls_all_FedAvg_KDFL, trn_perf_all_FedAvg_KDFL,
    tst_perf_all_FedAvg_KDFL] = train_RefKDFL(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                        epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                        model_func=model_KDFL, init_model=init_model_KD, TA_model=TA_model, save_period=save_period,
                                        lr_decay_per_round=lr_decay_per_round, opt=opt)

