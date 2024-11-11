import numpy as np
import torch
import torch.nn as nn
from utils.util_metrics import eval_2d, eval_2d_time
from utils.util_common import *
from utils.util_torch_fdm import gradient_xx_scalar, gradient_yy_scalar, gradient_t
from utils.util_loss import pointwise_rel_l2norm_loss, LpLoss
from einops import rearrange, repeat

def train_epoch(encoder,
                decoder,
                enc_optim,
                dec_optim,
                device,
                train_loader,
                train_params,
                dataset_params,
                net_params,
                epoch=None,
                time_forward=None,
                args=None):


    encoder.train()
    decoder.train()

    epoch_loss_dict = {}
    if train_params['is_pde_loss']:
        epoch_loss_dict['Loss_Phys'] = []
    if train_params['is_data_loss']:
        epoch_loss_dict['Loss_Data'] = []
    epoch_loss_dict['Loss_Total'] = []

    data_loss_type = train_params['data_loss_type'] if 'data_loss_type' in train_params.keys() else 'MSE'
    if data_loss_type == 'MSE':  # PDE Bench
        loss_data_fn = nn.MSELoss()
    elif data_loss_type == 'PRL2':  # Transformer paper
        loss_data_fn = pointwise_rel_l2norm_loss
    elif data_loss_type == 'LPL':  # PINO paper
        loss_data_fn = LpLoss()

    pde_loss_type = train_params['pde_loss_type'] if 'pde_loss_type' in train_params.keys() else 'MSE'
    if pde_loss_type == 'MSE':  # PDE Bench
        loss_pde_fn = nn.MSELoss()
    elif pde_loss_type == 'PRL2':  # Transformer paper
        loss_pde_fn = pointwise_rel_l2norm_loss
    elif pde_loss_type == 'LPL':  # PINO paper
        loss_pde_fn = LpLoss()

    if dataset_params['is_normalise']:
        if dataset_params['normaliser_type'] == 'gaussian':
            from utils.util_normaliser import GaussianNormalizer as Normalizer
        elif dataset_params['normaliser_type'] == 'range':
            from utils.util_normaliser import RangeNormalizer as Normalizer
        if dataset_params['normaliser_type'] == 'preset_gaussian':
            from utils.util_normaliser import GaussianNormalizer2 as Normalizer
        else:
            raise NotImplementedError

    epoch_metrics_dict = {}
    metrics_list = dataset_params['metrics_list']
    for metric_name in metrics_list:
        epoch_metrics_dict[metric_name] = []

    for step, (batch_input_x, batch_input_y, batch_solutions) in enumerate(train_loader):

        # move data to device
        batch_x = batch_input_x['feat']  # bs, res_grid, 3
        graph_x = batch_input_x['graph'] if 'graph' in batch_input_x.keys() else None
        batch_y = batch_input_y['feat']  # bs, res_grid, 2
        graph_y = batch_input_y['graph'] if 'graph' in batch_input_y.keys() else None
        batch_label = batch_solutions['u']

        u_mean = batch_solutions['u_mean'][0].to(device)
        u_std = batch_solutions['u_std'][0].to(device)

        input = batch_x
        input_pos = batch_x[:, :, -2:]
        propagate_pos = batch_y.clone()
        label = batch_label.clone()

        # move data to device
        input = input.to(device)
        input_graph = graph_x.to(device)
        input_pos = input_pos.to(device)
        propagate_pos = propagate_pos.to(device)
        label = label.to(device)

        if dataset_params['is_normalise']:
            u_normalizer = Normalizer(mean=u_mean, std=u_std)

            # label = u_normalizer.encode(label)
            input[..., :-2] = u_normalizer.encode(input[..., :-2])

        bs = batch_label.shape[0]

        z = encoder(h=input,
                    input_pos=input_pos,
                    g=input_graph)

        output_collection = decoder.rollout(h=z,
                                            input_pos=input_pos,
                                            propagate_pos=propagate_pos,
                                            forward_steps=time_forward)

        output = torch.concat(output_collection, dim=-2)

        if dataset_params['is_normalise']:
            # label = u_normalizer.encode(label)
            output = u_normalizer.decode(output)

        label_out_seq = label[..., dataset_params['in_seq']:dataset_params['in_seq'] + dataset_params['out_seq'], :]

        loss = 0

        # pde loss
        pde_loss_from_epoch = train_params['pde_loss_from_epoch'] if 'pde_loss_from_epoch' in train_params.keys() else 0
        if train_params['is_pde_loss']:
            raise NotImplementedError
            loss_pde = loss_sw_2d(loss_fn=loss_pde_fn,
                                  u=output,
                                  batchsize=bs,
                                  resolution=128 // dataset_params['reduced_resolution'],
                                  alpha=train_params['pde_loss_alpha'],
                                  beta=train_params['pde_loss_beta'], )
            loss += train_params['pde_loss_weight'] * loss_pde
            epoch_loss_dict['Loss_Phys'].append(loss_pde.detach().item())

        # data loss
        if train_params['is_data_loss']:
            loss_data = loss_data_fn(output, label_out_seq)
            loss += train_params['data_loss_weight'] * loss_data
            epoch_loss_dict['Loss_Data'].append(loss_data.detach().item())

        # total loss
        epoch_loss_dict['Loss_Total'].append(loss.detach().item())

        enc_optim.zero_grad()
        dec_optim.zero_grad()

        loss.backward()

        enc_optim.step()
        dec_optim.step()

        with torch.no_grad():
            train_params['evaluate_whole_seq'] = train_params['evaluate_whole_seq'] if 'evaluate_whole_seq' in train_params.keys() else True
            if train_params['evaluate_whole_seq']:
                output_full = label.clone()
                output_full[..., dataset_params['in_seq']:dataset_params['in_seq'] + dataset_params['out_seq'], :] = output
                step_eval_dict = eval_2d(output_full, label, metrics_list=metrics_list)
            else:
                step_eval_dict = eval_2d(output, label_out_seq, metrics_list=metrics_list)

        for metric_name in metrics_list:
            epoch_metrics_dict[metric_name].append(step_eval_dict[metric_name])

    epoch_loss_ave_dict = {}
    for loss_name in epoch_loss_dict.keys():
        epoch_loss_ave_dict[loss_name] = np.mean(epoch_loss_dict[loss_name])

    epoch_metrics_ave_dict = {}
    for metric_name in metrics_list:
        epoch_metrics_ave_dict[metric_name] = np.mean(epoch_metrics_dict[metric_name])

    return epoch_loss_ave_dict, epoch_metrics_ave_dict, enc_optim, dec_optim


def evaluate_epoch(encoder,
                   decoder,
                   device,
                   eval_loader,
                   train_params,
                   dataset_params,
                   net_params,
                   epoch=None,
                   eval_dir=None,
                   time_forward=None,
                   args=None):


    encoder.eval()
    decoder.eval()

    epoch_loss_dict = {}
    if train_params['is_pde_loss']:
        epoch_loss_dict['Loss_Phys'] = []
    if train_params['is_data_loss']:
        epoch_loss_dict['Loss_Data'] = []
    epoch_loss_dict['Loss_Total'] = []

    data_loss_type = train_params['data_loss_type'] if 'data_loss_type' in train_params.keys() else 'MSE'
    if data_loss_type == 'MSE':  # PDE Bench
        loss_data_fn = nn.MSELoss()
    elif data_loss_type == 'PRL2':  # Transformer paper
        loss_data_fn = pointwise_rel_l2norm_loss
    elif data_loss_type == 'LPL':  # PINO paper
        loss_data_fn = LpLoss()

    pde_loss_type = train_params['pde_loss_type'] if 'pde_loss_type' in train_params.keys() else 'MSE'
    if pde_loss_type == 'MSE':  # PDE Bench
        loss_pde_fn = nn.MSELoss()
    elif pde_loss_type == 'PRL2':  # Transformer paper
        loss_pde_fn = pointwise_rel_l2norm_loss
    elif pde_loss_type == 'LPL':  # PINO paper
        loss_pde_fn = LpLoss()

    if dataset_params['is_normalise']:
        if dataset_params['normaliser_type'] == 'gaussian':
            from utils.util_normaliser import GaussianNormalizer as Normalizer
        elif dataset_params['normaliser_type'] == 'range':
            from utils.util_normaliser import RangeNormalizer as Normalizer
        if dataset_params['normaliser_type'] == 'preset_gaussian':
            from utils.util_normaliser import GaussianNormalizer2 as Normalizer
        else:
            raise NotImplementedError

    epoch_metrics_dict = {}
    metrics_list = dataset_params['metrics_list']
    for metric_name in metrics_list:
        epoch_metrics_dict[metric_name] = []

    with torch.no_grad():
        for step, (batch_input_x, batch_input_y, batch_solutions) in enumerate(eval_loader):

            # move data to device
            batch_x = batch_input_x['feat']  # bs, res_grid, 3
            graph_x = batch_input_x['graph'] if 'graph' in batch_input_x.keys() else None
            batch_y = batch_input_y['feat']  # bs, res_grid, 2
            graph_y = batch_input_y['graph'] if 'graph' in batch_input_y.keys() else None
            batch_label = batch_solutions['u']

            u_mean = batch_solutions['u_mean'][0].to(device)
            u_std = batch_solutions['u_std'][0].to(device)

            input = batch_x
            input_pos = batch_x[:, :, -2:]
            propagate_pos = batch_y.clone()
            label = batch_label.clone()

            # move data to device
            input = input.to(device)
            input_graph = graph_x.to(device)
            input_pos = input_pos.to(device)
            propagate_pos = propagate_pos.to(device)
            label = label.to(device)

            if dataset_params['is_normalise']:
                u_normalizer = Normalizer(mean=u_mean, std=u_std)

                # label = u_normalizer.encode(label)
                input[..., :-2] = u_normalizer.encode(input[..., :-2])

            bs = batch_label.shape[0]

            z = encoder(h=input,
                        input_pos=input_pos,
                        g=input_graph)

            output_collection = decoder.rollout(h=z,
                                                input_pos=input_pos,
                                                propagate_pos=propagate_pos,
                                                forward_steps=time_forward)

            output = torch.concat(output_collection, dim=-2)

            if dataset_params['is_normalise']:
                # label = u_normalizer.encode(label)
                output = u_normalizer.decode(output)

            label_out_seq = label[..., dataset_params['in_seq']:dataset_params['in_seq'] + dataset_params['out_seq'], :]

            loss = 0

            # pde loss
            pde_loss_from_epoch = train_params['pde_loss_from_epoch'] if 'pde_loss_from_epoch' in train_params.keys() else 0
            if train_params['is_pde_loss']:
                raise NotImplementedError
                loss_pde = loss_sw_2d(loss_fn=loss_pde_fn,
                                      u=output,
                                      batchsize=bs,
                                      resolution=128 // dataset_params['reduced_resolution'],
                                      alpha=train_params['pde_loss_alpha'],
                                      beta=train_params['pde_loss_beta'], )
                loss += train_params['pde_loss_weight'] * loss_pde
                epoch_loss_dict['Loss_Phys'].append(loss_pde.detach().item())

            # data loss
            if train_params['is_data_loss']:
                loss_data = loss_data_fn(output, label_out_seq)
                loss += train_params['data_loss_weight'] * loss_data
                epoch_loss_dict['Loss_Data'].append(loss_data.detach().item())

            # total loss
            epoch_loss_dict['Loss_Total'].append(loss.detach().item())

            train_params['evaluate_whole_seq'] = train_params['evaluate_whole_seq'] if 'evaluate_whole_seq' in train_params.keys() else True
            if train_params['evaluate_whole_seq']:
                output_full = label.clone()
                output_full[..., dataset_params['in_seq']:dataset_params['in_seq'] + dataset_params['out_seq'], :] = output
                step_eval_dict = eval_2d(output_full, label, metrics_list=metrics_list)
            else:
                step_eval_dict = eval_2d(output, label_out_seq, metrics_list=metrics_list)

            for metric_name in metrics_list:
                epoch_metrics_dict[metric_name].append(step_eval_dict[metric_name])

        epoch_loss_ave_dict = {}
        for loss_name in epoch_loss_dict.keys():
            epoch_loss_ave_dict[loss_name] = np.mean(epoch_loss_dict[loss_name])

        epoch_metrics_ave_dict = {}
        for metric_name in metrics_list:
            epoch_metrics_ave_dict[metric_name] = np.mean(epoch_metrics_dict[metric_name])

    return epoch_loss_ave_dict, epoch_metrics_ave_dict


def evaluate_1step(encoder,
                   decoder,
                   device,
                   eval_loader,
                   train_params,
                   dataset_params,
                   net_params,
                   epoch=None,
                   eval_dir=None,
                   time_forward=None,
                   args=None):


    encoder.eval()
    decoder.eval()

    epoch_loss_dict = {}
    if train_params['is_pde_loss']:
        epoch_loss_dict['Loss_Phys'] = []
    if train_params['is_data_loss']:
        epoch_loss_dict['Loss_Data'] = []
    epoch_loss_dict['Loss_Total'] = []

    data_loss_type = train_params['data_loss_type'] if 'data_loss_type' in train_params.keys() else 'MSE'
    if data_loss_type == 'MSE':  # PDE Bench
        loss_data_fn = nn.MSELoss()
    elif data_loss_type == 'PRL2':  # Transformer paper
        loss_data_fn = pointwise_rel_l2norm_loss
    elif data_loss_type == 'LPL':  # PINO paper
        loss_data_fn = LpLoss()

    pde_loss_type = train_params['pde_loss_type'] if 'pde_loss_type' in train_params.keys() else 'MSE'
    if pde_loss_type == 'MSE':  # PDE Bench
        loss_pde_fn = nn.MSELoss()
    elif pde_loss_type == 'PRL2':  # Transformer paper
        loss_pde_fn = pointwise_rel_l2norm_loss
    elif pde_loss_type == 'LPL':  # PINO paper
        loss_pde_fn = LpLoss()

    if dataset_params['is_normalise']:
        if dataset_params['normaliser_type'] == 'gaussian':
            from utils.util_normaliser import GaussianNormalizer as Normalizer
        elif dataset_params['normaliser_type'] == 'range':
            from utils.util_normaliser import RangeNormalizer as Normalizer
        if dataset_params['normaliser_type'] == 'preset_gaussian':
            from utils.util_normaliser import GaussianNormalizer2 as Normalizer
        else:
            raise NotImplementedError

    epoch_metrics_dict = {}
    metrics_list = dataset_params['metrics_list']
    for metric_name in metrics_list:
        epoch_metrics_dict[metric_name] = []

    with torch.no_grad():
        for step, (batch_input_x, batch_input_y, batch_solutions) in enumerate(eval_loader):

            if step > 0:
                break

            # move data to device
            batch_x = batch_input_x['feat']  # bs, res_grid, 3
            graph_x = batch_input_x['graph'] if 'graph' in batch_input_x.keys() else None
            batch_y = batch_input_y['feat']  # bs, res_grid, 2
            graph_y = batch_input_y['graph'] if 'graph' in batch_input_y.keys() else None
            batch_label = batch_solutions['u']

            u_mean = batch_solutions['u_mean'][0].to(device)
            u_std = batch_solutions['u_std'][0].to(device)

            input = batch_x
            input_pos = batch_x[:, :, -2:]
            propagate_pos = batch_y.clone()
            label = batch_label.clone()

            # move data to device
            input = input.to(device)
            input_graph = graph_x.to(device)
            input_pos = input_pos.to(device)
            propagate_pos = propagate_pos.to(device)
            label = label.to(device)

            if dataset_params['is_normalise']:
                u_normalizer = Normalizer(mean=u_mean, std=u_std)

                # label = u_normalizer.encode(label)
                input[..., :-2] = u_normalizer.encode(input[..., :-2])

            bs = batch_label.shape[0]

            z = encoder(h=input,
                        input_pos=input_pos,
                        g=input_graph)

            output_collection = decoder.rollout(h=z,
                                                input_pos=input_pos,
                                                propagate_pos=propagate_pos,
                                                forward_steps=time_forward)

            output = torch.concat(output_collection, dim=-2)

            if dataset_params['is_normalise']:
                # label = u_normalizer.encode(label)
                output = u_normalizer.decode(output)

            label_out_seq = label[..., dataset_params['in_seq']:dataset_params['in_seq'] + dataset_params['out_seq'], :]

            loss = 0

            # pde loss
            pde_loss_from_epoch = train_params['pde_loss_from_epoch'] if 'pde_loss_from_epoch' in train_params.keys() else 0
            if train_params['is_pde_loss']:
                raise NotImplementedError
                loss_pde = loss_sw_2d(loss_fn=loss_pde_fn,
                                      u=output,
                                      batchsize=bs,
                                      resolution=128 // dataset_params['reduced_resolution'],
                                      alpha=train_params['pde_loss_alpha'],
                                      beta=train_params['pde_loss_beta'], )
                loss += train_params['pde_loss_weight'] * loss_pde
                epoch_loss_dict['Loss_Phys'].append(loss_pde.detach().item())

            # data loss
            if train_params['is_data_loss']:
                loss_data = loss_data_fn(output, label_out_seq)
                loss += train_params['data_loss_weight'] * loss_data
                epoch_loss_dict['Loss_Data'].append(loss_data.detach().item())

            # total loss
            epoch_loss_dict['Loss_Total'].append(loss.detach().item())

            train_params['evaluate_whole_seq'] = train_params['evaluate_whole_seq'] if 'evaluate_whole_seq' in train_params.keys() else True
            if train_params['evaluate_whole_seq']:
                output_full = label.clone()
                output_full[..., dataset_params['in_seq']:dataset_params['in_seq'] + dataset_params['out_seq'], :] = output
                step_eval_dict = eval_2d(output_full, label, metrics_list=metrics_list)
            else:
                step_eval_dict = eval_2d(output, label_out_seq, metrics_list=metrics_list)

            for metric_name in metrics_list:
                epoch_metrics_dict[metric_name].append(step_eval_dict[metric_name])

            # save array
            dataset_params['resolution'] = 128 // dataset_params['reduced_resolution']
            gt_array = label.cpu().numpy().reshape(bs, dataset_params['resolution'], dataset_params['resolution'], 101)  # (bs, h, w, 1)
            in_seq_array = input.cpu().numpy().reshape(bs, dataset_params['resolution'], dataset_params['resolution'], 12,)  # (bs, h, w, 3)
            pred_array = output_full.cpu().numpy().reshape(bs, dataset_params['resolution'], dataset_params['resolution'], 101)  # (bs, h, w, 1)

            result_path = os.path.join(eval_dir)
            mkdir(result_path)
            np.savez(os.path.join(result_path, 'results_shallow_water_vis.npz'),
                     gt=gt_array,
                     in_seq=in_seq_array,
                     pred=pred_array)

        epoch_loss_ave_dict = {}
        for loss_name in epoch_loss_dict.keys():
            epoch_loss_ave_dict[loss_name] = np.mean(epoch_loss_dict[loss_name])

        epoch_metrics_ave_dict = {}
        for metric_name in metrics_list:
            epoch_metrics_ave_dict[metric_name] = np.mean(epoch_metrics_dict[metric_name])

    return epoch_loss_ave_dict, epoch_metrics_ave_dict

def loss_sw_2d(loss_fn, uv, batchsize, resolution, space_range=2, time_range=5, alpha=0, beta=0):
    raise NotImplementedError

    return loss
