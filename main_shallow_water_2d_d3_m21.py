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
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.utils.data import DataLoader

from einops import rearrange
from tensorboardX import SummaryWriter
from tqdm import tqdm

from nets.load_net import load_net
from data_pde.load_data import load_data
from train.train_shallow_water_2d_d3_m21 import train_epoch, evaluate_epoch

from utils.util_common import *

import wandb

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


def train_val_pipeline(MODEL_NAME, dataset, params, out_dir):
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
    trainset, valset = dataset.train, dataset.val
    # trainset, valset, testset = dataset.train, dataset.val, dataset.test

    print("Training: ", len(trainset))
    print("Validation: ", len(valset))

    # path
    log_dir = os.path.join(out_dir, 'log')
    ckpt_dir = os.path.join(out_dir, 'checkpoint')
    eval_dir = os.path.join(out_dir, 'eval')
    mkdir(log_dir)
    mkdir(ckpt_dir)
    mkdir(eval_dir)

    # logger
    writer = SummaryWriter(log_dir=os.path.join(log_dir))

    # wandb logger
    wandb.define_metric("Epoch")
    wandb.define_metric('Learning Rate/Encoder', step_metric="Epoch")
    wandb.define_metric('Learning Rate/Decoder', step_metric="Epoch")
    wandb.define_metric('TRAIN LOSS/Loss_MSE', step_metric="Epoch")
    wandb.define_metric('VAL LOSS/Loss_MSE', step_metric="Epoch")
    for metric_name in dataset_params['train']['metrics_list']:
        wandb.define_metric(f'TRAIN METRICS/{metric_name}', step_metric="Epoch")
    for metric_name in dataset_params['val']['metrics_list']:
        wandb.define_metric(f'VAL METRICS/{metric_name}', step_metric="Epoch")

    # load model
    encoder, decoder = build_model(MODEL_NAME, net_params, device)

    # record model
    wandb.watch(encoder)
    wandb.watch(decoder)

    # optimiser
    enc_optim = torch.optim.Adam(list(encoder.parameters()), lr=train_params['init_lr'], weight_decay=train_params['weight_decay'])
    dec_optim = torch.optim.Adam(list(decoder.parameters()), lr=train_params['init_lr'], weight_decay=train_params['weight_decay'])

    if params['checkpoint_path']:
        pretrained_checkpoint = torch.load(params['checkpoint_path'])
        encoder.load_state_dict(pretrained_checkpoint['encoder'], strict=True)
        decoder.load_state_dict(pretrained_checkpoint['decoder'], strict=True)
        enc_optim.load_state_dict(pretrained_checkpoint['enc_optim'])
        dec_optim.load_state_dict(pretrained_checkpoint['dec_optim'])
        start_n_epochs = pretrained_checkpoint['epoch']
        print('Loading pretrain model... Training from epoch {}'.format(start_n_epochs))

    else:
        start_n_epochs = 0
        print('Training from scratch...')

    # learning rate schedule
    lr_schedule = 'ReduceLROnPlateau' if 'lr_schedule' not in train_params.keys() else train_params['lr_schedule']
    if lr_schedule == 'ReduceLROnPlateau':
        enc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(enc_optim,
                                                             mode='min',
                                                             factor=train_params['lr_reduce_factor'],
                                                             patience=train_params['lr_schedule_patience'],
                                                             min_lr=train_params['min_lr'],
                                                             verbose=True)
        dec_scheduler = optim.lr_scheduler.ReduceLROnPlateau(dec_optim,
                                                             mode='min',
                                                             factor=train_params['lr_reduce_factor'],
                                                             patience=train_params['lr_schedule_patience'],
                                                             min_lr=train_params['min_lr'],
                                                             verbose=True)
    elif lr_schedule == 'OneCycleLR':
        enc_scheduler = OneCycleLR(enc_optim,
                                   max_lr=train_params['init_lr'],
                                   total_steps=train_params['epochs'],  # epoch --> step in the schedule
                                   **{k: train_params[k] for k in ['div_factor', 'pct_start', 'final_div_factor'] if k in train_params}
                                   )
        dec_scheduler = OneCycleLR(dec_optim,
                                   max_lr=train_params['init_lr'],
                                   total_steps=train_params['epochs'],  # epoch --> step in the schedule
                                   **{k: train_params[k] for k in ['div_factor', 'pct_start', 'final_div_factor'] if k in train_params}
                                   )

    if start_n_epochs > 0:
        enc_scheduler.step(start_n_epochs - 1)
        dec_scheduler.step(start_n_epochs - 1)

    # record loss and accuracy
    epoch_train_eval_dict = {}
    for metric_name in dataset_params['train']['metrics_list']:
        epoch_train_eval_dict[metric_name] = []
    epoch_val_eval_dict = {}
    for metric_name in dataset_params['val']['metrics_list']:
        epoch_val_eval_dict[metric_name] = []

    # data loader
    train_loader = DataLoader(trainset,
                              batch_size=dataset_params['train']['batch_size'],
                              shuffle=True,
                              num_workers=dataset_params['train']['batch_size'],
                              drop_last=True,
                              pin_memory=False,
                              collate_fn=dataset.collate)
    val_loader = DataLoader(valset,
                            batch_size=dataset_params['val']['batch_size'],
                            shuffle=False,
                            num_workers=dataset_params['val']['batch_size'],
                            drop_last=False,
                            pin_memory=False,
                            collate_fn=dataset.collate)

    for epoch in range(train_params['epochs']):

        current_epoch = epoch + start_n_epochs

        epoch_start_time = time.time()

        epoch_train_loss_ave_dict, epoch_train_metric_ave_dict, enc_optim, dec_optim = train_epoch(encoder=encoder,
                                                                                                   decoder=decoder,
                                                                                                   enc_optim=enc_optim,
                                                                                                   dec_optim=dec_optim,
                                                                                                   device=device,
                                                                                                   train_loader=train_loader,
                                                                                                   train_params=train_params,
                                                                                                   dataset_params=dataset_params['train'],
                                                                                                   net_params=net_params,
                                                                                                   epoch=current_epoch,
                                                                                                   time_forward=dataset_params['train']['out_seq'],
                                                                                                   )

        epoch_val_loss_ave_dict, epoch_val_metric_ave_dict = evaluate_epoch(encoder=encoder,
                                                                            decoder=decoder,
                                                                            device=device,
                                                                            eval_loader=val_loader,
                                                                            train_params=train_params,
                                                                            dataset_params=dataset_params['val'],
                                                                            net_params=net_params,
                                                                            epoch=current_epoch,
                                                                            eval_dir=eval_dir,
                                                                            time_forward=dataset_params['val']['out_seq'],
                                                                            )

        current_enc_lr = enc_optim.param_groups[0]['lr']
        current_dec_lr = dec_optim.param_groups[0]['lr']
        enc_scheduler.step(current_epoch)
        dec_scheduler.step(current_epoch)

        # record loss and accuracy
        log_wandb = {'Epoch': current_epoch}

        writer.add_scalar('Learning Rate/Encoder', current_enc_lr, current_epoch)
        writer.add_scalar('Learning Rate/Decoder', current_dec_lr, current_epoch)
        log_wandb['Learning Rate/Encoder'] = current_enc_lr
        log_wandb['Learning Rate/Decoder'] = current_dec_lr

        for loss_name in epoch_train_loss_ave_dict.keys():
            writer.add_scalar(f'TRAIN LOSS/{loss_name}', epoch_train_loss_ave_dict[loss_name], current_epoch)
            log_wandb[f'TRAIN LOSS/{loss_name}'] = epoch_train_loss_ave_dict[loss_name]

        for loss_name in epoch_val_loss_ave_dict.keys():
            writer.add_scalar(f'VAL LOSS/{loss_name}', epoch_val_loss_ave_dict[loss_name], current_epoch)
            log_wandb[f'VAL LOSS/{loss_name}'] = epoch_val_loss_ave_dict[loss_name]

        for metric_name in dataset_params['train']['metrics_list']:
            epoch_train_eval_dict[metric_name].append(epoch_train_metric_ave_dict[metric_name])
            writer.add_scalar(f'TRAIN METRICS/{metric_name}', epoch_train_metric_ave_dict[metric_name], current_epoch)
            log_wandb[f'TRAIN METRICS/{metric_name}'] = epoch_train_metric_ave_dict[metric_name]

        for metric_name in dataset_params['val']['metrics_list']:
            epoch_val_eval_dict[metric_name].append(epoch_val_metric_ave_dict[metric_name])
            writer.add_scalar(f'VAL METRICS/{metric_name}', epoch_val_metric_ave_dict[metric_name], current_epoch)
            log_wandb[f'VAL METRICS/{metric_name}'] = epoch_val_metric_ave_dict[metric_name]

        wandb.log(log_wandb)

        # record time
        epoch_end_time = time.time()
        epoch_time_used = epoch_end_time - epoch_start_time
        total_time_used = epoch_end_time - initial_start_time
        per_epoch_time.append(epoch_time_used)

        # print log
        print('Epoch {:03d}/{:03d} - Epoch Time Used: {:.3f}s; Total Time Used: {:.3f}s; Train Loss: {:.3f}; Val Loss: {:.3f}; '
              .format(current_epoch, train_params['epochs'], epoch_time_used, total_time_used, epoch_train_loss_ave_dict['Loss_Total'], epoch_val_loss_ave_dict['Loss_Total']))

        # metric_log = 'Epoch {:03d}/{:03d} - '.format(current_epoch, train_params['epochs'])
        # for metric_name in dataset_params['train']['metrics_list']:
        #     metric_log += 'Train {}: {:.3f}; '.format(metric_name, epoch_train_metric_ave_dict[metric_name])
        # for metric_name in dataset_params['val']['metrics_list']:
        #     metric_log += 'Val {}: {:.3f}; '.format(metric_name, epoch_val_metric_ave_dict[metric_name])
        # print(metric_log)

        metric_log = 'Epoch {:03d}/{:03d} - '.format(current_epoch, train_params['epochs'])
        for metric_name in dataset_params['train']['metrics_list']:
            metric_log += 'Train {}: {}; '.format(metric_name, epoch_train_metric_ave_dict[metric_name])
        for metric_name in dataset_params['val']['metrics_list']:
            metric_log += 'Val {}: {}; '.format(metric_name, epoch_val_metric_ave_dict[metric_name])
        print(metric_log)

        if params["is_save_model"] and not (current_epoch % params["save_model_every"]):
            # save checkpoint
            checkpoint = {
                'epoch': current_epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'enc_optim': enc_optim.state_dict(),
                'dec_optim': dec_optim.state_dict(),
            }
            torch.save(checkpoint, os.path.join(ckpt_dir, 'checkpoint_epoch_{}.pth'.format(current_epoch)))

            # # delete previous weight
            # files = glob.glob(ckpt_dir + '/*.pth')
            # for file in files:
            #     epoch_nb = file.split('_')[-1]
            #     epoch_nb = int(epoch_nb.split('.')[0])
            #     if epoch_nb < current_epoch - 1:
            #         os.remove(file)

        # early stopping
        # # TODO: More early stopping condition
        # if enc_optim.param_groups[0]['lr'] < train_params['min_lr']:
        #     # print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
        #     # break
        #     enc_optim.param_groups[0]['lr'] = train_params['min_lr']
        # if dec_optim.param_groups[0]['lr'] < train_params['min_lr']:
        #     # print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
        #     # break
        #     dec_optim.param_groups[0]['lr'] = train_params['min_lr']

    writer.close()




if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # load configuration from json
    parser = argparse.ArgumentParser("TRAIN THE PDE GRAPH TRANSFORMER")
    parser.add_argument('--config', type=str, default="config/ShallowWater2D/PDEBench/config_SW2D_PDEBench_M21b_D3_PNG_Customx1_Customx1_Tx1_RelPE_InitO_BN_MSE_OCLR.json", help="json configuration file")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # set device and random seed
    device = peripheral_setup(gpu_list=config['device'], seed=config['seed'],)

    # set wandb logger
    os.environ['WANDB_MODE'] = config['wandb']['mode']
    wandb.init(project=config['wandb']['project_name'], entity="x")
    if config['wandb']['is_sweep']:
        for key in wandb.config.keys():
            print(f'SWEEP PARA -- {key}: {wandb.config[key]}')

            # config['dataset_params']['train'] = wandb.config['batch_size']
            # config['dataset_params']['val'] = wandb.config['batch_size']
            # config['train_params']['init_lr'] = wandb.config['init_lr']
            # config['train_params']['init_type'] = wandb.config['init_type']

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

    # set timestamp
    time_stamp = time.strftime('%Y%m%d%H%M%S')
    if config['wandb']['is_sweep']:
        time_stamp = 'SWEEP'
    if 'DEBUG' in config['model_name']:
        time_stamp = 'DEBUG'
    config['time_stamp'] = time_stamp

    # wandb record config
    wandb.config.update(config)

    # load dataset
    dataset = load_data(DATASET_NAME, config['dataset_params'])

    # set dir
    out_dir = os.path.join(out_dir, '{}'.format(PROJECT_NAME), 'RUN_{}'.format(time_stamp),)
    mkdir(out_dir)

    # train & val
    train_val_pipeline(MODEL_NAME, dataset, config, out_dir)
