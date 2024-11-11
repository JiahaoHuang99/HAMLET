import os

import dgl
import shutil
import logging
import time
import argparse
import json
import random
import glob
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from einops import rearrange
from tensorboardX import SummaryWriter
from tqdm import tqdm

from nets.load_net import load_net
from data_pde.load_data import load_data
from train.train_darcy_flow_d3_m21 import evaluate_1step

from utils.util_common import *


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

"""
    GPU SETUP
"""


def peripheral_setup(gpu_list, seed):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    assert isinstance(gpu_list, list), ValueError('gpu_list should be a list.')

    device = 'cuda' if gpu_list else 'cpu'
    # if torch.cuda.is_available() and device == 'cuda':
    if device == 'cuda':
        gpu_list = ','.join(str(x) for x in gpu_list)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True  # type:ignore
        torch.backends.cudnn.deterministic = True  # type:ignore
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.autograd.set_detect_anomaly(True)
        print("Using CUDA...")
        print("GPU number: {}".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            print("GPU {}: {}".format(i, torch.cuda.get_device_name(i)))
    else:
        print("Using CPU, GPU not available..")
    device = torch.device(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    return device


"""
    Viewing model config and params
"""


def view_model_param(MODEL_NAME, model_params):
    encoder, decoder = load_net(MODEL_NAME, model_params)
    total_params = 0
    total_params += sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    print('Total parameters:', total_params)
    return total_params


def build_model(MODEL_NAME, model_params, device):
    encoder, decoder = load_net(MODEL_NAME, model_params)
    encoder.to(device)
    decoder.to(device)

    return encoder, decoder


def val_pipeline(MODEL_NAME, dataset, params, out_dir):
    initial_start_time = time.time()
    per_epoch_time = []

    # parameter
    train_params = params['train_params']
    dataset_params = params['dataset_params']
    net_params = params['net_params']

    # device
    device = net_params['device']

    # dataset
    DATASET_NAME = dataset.name

    # dataset
    valset = dataset.val

    print("Validation Graphs: ", len(valset))

    # load model
    encoder, decoder = build_model(MODEL_NAME, net_params, device)

    # optimiser
    enc_optim = torch.optim.Adam(list(encoder.parameters()), lr=train_params['init_lr'], weight_decay=train_params['weight_decay'])
    dec_optim = torch.optim.Adam(list(decoder.parameters()), lr=train_params['init_lr'], weight_decay=train_params['weight_decay'])

    assert params['checkpoint_path']
    pretrained_checkpoint = torch.load(params['checkpoint_path'])
    encoder.load_state_dict(pretrained_checkpoint['encoder'], strict=True)
    decoder.load_state_dict(pretrained_checkpoint['decoder'], strict=True)
    enc_optim.load_state_dict(pretrained_checkpoint['enc_optim'])
    dec_optim.load_state_dict(pretrained_checkpoint['dec_optim'])
    start_n_epochs = pretrained_checkpoint['epoch']
    print('Loading pretrain model... Training from epoch {}'.format(start_n_epochs))

    # record loss and accuracy
    epoch_val_eval_dict = {}
    for metric_name in dataset_params['val']['metrics_list']:
        epoch_val_eval_dict[metric_name] = []

    # data loader
    val_loader = DataLoader(valset,
                            batch_size=dataset_params['val']['batch_size'],
                            shuffle=False,
                            num_workers=dataset_params['val']['batch_size'],
                            drop_last=False,
                            pin_memory=False,
                            collate_fn=dataset.collate)

    epoch_start_time = time.time()

    epoch_val_loss_ave_dict, epoch_val_metric_ave_dict = evaluate_1step(encoder=encoder,
                                                                        decoder=decoder,
                                                                        device=device,
                                                                        eval_loader=val_loader,
                                                                        train_params=train_params,
                                                                        dataset_params=dataset_params['val'],
                                                                        net_params=net_params,
                                                                        epoch=None,
                                                                        eval_dir=os.path.join(out_dir, 'array'),
                                                                        time_forward=None,
                                                                        )

    # record time
    epoch_end_time = time.time()
    epoch_time_used = epoch_end_time - epoch_start_time
    total_time_used = epoch_end_time - initial_start_time
    per_epoch_time.append(epoch_time_used)

    # print log
    print('Time Used: {:.3f}s; Total Time Used: {:.3f}s; Val Loss: {:.3f}; '
          .format(epoch_time_used, total_time_used, epoch_val_loss_ave_dict['Loss_Total']))

    metric_log = ''
    for metric_name in dataset_params['val']['metrics_list']:
        metric_log += 'Val {}: {}; '.format(metric_name, epoch_val_metric_ave_dict[metric_name])
    print(metric_log)



if __name__ == '__main__':

    set_random_seed(0)

    # load configuration from json
    parser = argparse.ArgumentParser("TRAIN THE PDE GRAPH TRANSFORMER")
    parser.add_argument('--config', type=str, default="config/DarcyFlow/PDEBench/test/config_test_DarcyFlow_PDEBench_M21b_D3_beta1.0_Stanx4_Full_r0.1_RelPE_InitO_BN_MSE_OCLR1.json", help="json configuration file")
    parser.add_argument('--weight_path', type=str, default=None,)

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # set device and random seed
    device = peripheral_setup(gpu_list=config['device'], seed=config['seed'],)

    # parameter
    PROJECT_NAME = config['project_name']
    MODEL_NAME = config['model_name']
    DATASET_NAME = config['dataset_name']
    out_dir = config['out_dir']

    config['train_params']['model'] = MODEL_NAME
    config['train_params']['dataset'] = DATASET_NAME
    config['dataset_params']['model'] = MODEL_NAME
    config['dataset_params']['dataset'] = DATASET_NAME
    config['net_params']['model'] = MODEL_NAME
    config['net_params']['dataset'] = DATASET_NAME
    config['net_params']['device'] = device
    config['net_params']['gpu_id'] = config['device']
    config['net_params']['total_param'] = view_model_param(MODEL_NAME, config['net_params'])
    if args.weight_path is not None:
        config['checkpoint_path'] = args.weight_path

    # set timestamp
    # time_stamp = time.strftime('%Y%m%d%H%M%S')
    time_stamp = 'TEST'
    config['time_stamp'] = time_stamp

    # load dataset
    dataset = load_data(DATASET_NAME, config['dataset_params'])

    # set dir
    out_dir = os.path.join(out_dir, '{}'.format(PROJECT_NAME), 'RUN_{}'.format(time_stamp),)
    mkdir(out_dir)

    # train & val
    val_pipeline(MODEL_NAME, dataset, config, out_dir)
