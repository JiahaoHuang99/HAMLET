import numpy as np
import torch
import torch.nn as nn
from utils.util_metrics import eval_2d
from utils.util_common import *
from utils.util_torch_fdm import gradient_xy_vector, gradient_xy_scalar, gradient_t, laplacian
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

    if dataset_params['is_normalise']:
        if dataset_params['normaliser_type'] == 'gaussian':
            from utils.util_normaliser import GaussianNormalizer as Normalizer
        elif dataset_params['normaliser_type'] == 'range':
            from utils.util_normaliser import RangeNormalizer as Normalizer
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
        batch_label = batch_solutions['u']  # bs, res_grid, 1
        batch_label_a = batch_solutions['a']  # bs, res_grid, 1

        input = batch_x
        input_pos = batch_x[:, :, -2:]
        propagate_pos = batch_y.clone()
        label = batch_label.clone()
        label_a = batch_label_a.clone()

        # move data to device
        input = input.to(device)
        input_graph = graph_x.to(device)
        input_pos = input_pos.to(device)
        propagate_pos = propagate_pos.to(device)
        label = label.to(device)
        label_a = label_a.to(device)

        if dataset_params['is_normalise']:
            raise NotImplementedError

        bs = batch_label.shape[0]

        z = encoder(h=input,
                    input_pos=input_pos,
                    g=input_graph)

        output = decoder(h=z,
                         input_pos=input_pos,
                         propagate_pos=propagate_pos,)

        if dataset_params['is_normalise']:
            raise NotImplementedError

        loss = 0

        # pde loss
        pde_loss_type = train_params['pde_loss_type'] if 'pde_loss_type' in train_params.keys() else None
        pde_loss_from_epoch = train_params['pde_loss_from_epoch'] if 'pde_loss_from_epoch' in train_params.keys() else 0
        if train_params['is_pde_loss']:
            if epoch >= pde_loss_from_epoch:
                # FIXME: add PDE loss
                loss_pde = loss_darcy_flow(u=output,
                                           a=label_a,
                                           force=dataset_params['beta'],
                                           batchsize=bs,
                                           resolution=128 // dataset_params['reduced_resolution'],
                                           alpha=train_params['pde_loss_alpha'],
                                           beta=train_params['pde_loss_beta'],
                                           type=pde_loss_type,)
                loss += train_params['pde_loss_weight'] * loss_pde
                epoch_loss_dict['Loss_Phys'].append(loss_pde.detach().item())
            else:
                epoch_loss_dict['Loss_Phys'].append(0)

        # data loss
        if train_params['is_data_loss']:
            loss_data = loss_data_fn(output, label)
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
            step_eval_dict = eval_2d(output, label, metrics_list=metrics_list)

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

    if dataset_params['is_normalise']:
        if dataset_params['normaliser_type'] == 'gaussian':
            from utils.util_normaliser import GaussianNormalizer as Normalizer
        elif dataset_params['normaliser_type'] == 'range':
            from utils.util_normaliser import RangeNormalizer as Normalizer
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
            batch_label = batch_solutions['u']  # bs, res_grid, 1
            batch_label_a = batch_solutions['a']  # bs, res_grid, 1

            input = batch_x
            input_pos = batch_x[:, :, -2:]
            propagate_pos = batch_y.clone()
            label = batch_label.clone()
            label_a = batch_label_a.clone()

            # move data to device
            input = input.to(device)
            input_graph = graph_x.to(device)
            input_pos = input_pos.to(device)
            propagate_pos = propagate_pos.to(device)
            label = label.to(device)
            label_a = label_a.to(device)

            if dataset_params['is_normalise']:
                input_normalizer = Normalizer(input)
                input = input_normalizer.encode(input)

                label_normalizer = Normalizer(label)
                label = label_normalizer.encode(label)

            bs = batch_label.shape[0]

            z = encoder(h=input,
                        input_pos=input_pos,
                        g=input_graph)

            output = decoder(h=z,
                             input_pos=input_pos,
                             propagate_pos=propagate_pos,)

            loss = 0

            # pde loss
            pde_loss_type = train_params['pde_loss_type'] if 'pde_loss_type' in train_params.keys() else None
            pde_loss_from_epoch = train_params['pde_loss_from_epoch'] if 'pde_loss_from_epoch' in train_params.keys() else 0
            if train_params['is_pde_loss']:
                if epoch >= pde_loss_from_epoch:
                    # FIXME: add PDE loss
                    loss_pde = loss_darcy_flow(u=output,
                                               a=label_a,
                                               force=dataset_params['beta'],
                                               batchsize=bs,
                                               resolution=128 // dataset_params['reduced_resolution'],
                                               alpha=train_params['pde_loss_alpha'],
                                               beta=train_params['pde_loss_beta'],
                                               type=pde_loss_type)

                    loss += train_params['pde_loss_weight'] * loss_pde
                    epoch_loss_dict['Loss_Phys'].append(loss_pde.detach().item())
                else:
                    epoch_loss_dict['Loss_Phys'].append(0)

            # data loss
            if train_params['is_data_loss']:
                loss_data = loss_data_fn(output, label)
                loss += train_params['data_loss_weight'] * loss_data
                epoch_loss_dict['Loss_Data'].append(loss_data.detach().item())

            # total loss
            epoch_loss_dict['Loss_Total'].append(loss.detach().item())

            if dataset_params['is_normalise']:
                output = label_normalizer.decode(output)
                label = label_normalizer.decode(label)

            # import torch.nn.functional as F
            # output = rearrange(output, 'b (h w) c -> b c h w', h=141)
            # output = output[..., 1:-1, 1:-1].contiguous()
            # output = F.pad(output, (1, 1, 1, 1), "constant", 0)
            # output = rearrange(output, 'b c h w -> b (h w) c')

            step_eval_dict = eval_2d(output, label, metrics_list=metrics_list)
            for metric_name in metrics_list:
                epoch_metrics_dict[metric_name].append(step_eval_dict[metric_name])

            # save array
            # output_array = output.detach().cpu().numpy().reshape(bs, -1, 1)  # (bs, n_sample, t=1)
            #
            # for idx_bs in range(bs):
            #     sample_idx = idx_bs + step * bs
            #     result_path = os.path.join(eval_dir, 'epoch_{}'.format(epoch))
            #     mkdir(result_path)
            #     np.save(os.path.join(result_path, 'sample_{}.npy'.format(sample_idx)), output_array[idx_bs, ...])

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

    if dataset_params['is_normalise']:
        if dataset_params['normaliser_type'] == 'gaussian':
            from utils.util_normaliser import GaussianNormalizer as Normalizer
        elif dataset_params['normaliser_type'] == 'range':
            from utils.util_normaliser import RangeNormalizer as Normalizer
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
            batch_label = batch_solutions['u']  # bs, res_grid, 1
            batch_label_a = batch_solutions['a']  # bs, res_grid, 1

            input = batch_x
            input_pos = batch_x[:, :, -2:]
            propagate_pos = batch_y.clone()
            label = batch_label.clone()
            label_a = batch_label_a.clone()

            # move data to device
            input = input.to(device)
            input_graph = graph_x.to(device)
            input_pos = input_pos.to(device)
            propagate_pos = propagate_pos.to(device)
            label = label.to(device)
            label_a = label_a.to(device)

            if dataset_params['is_normalise']:
                input_normalizer = Normalizer(input)
                input = input_normalizer.encode(input)

                label_normalizer = Normalizer(label)
                label = label_normalizer.encode(label)

            bs = batch_label.shape[0]

            z = encoder(h=input,
                        input_pos=input_pos,
                        g=input_graph)

            output = decoder(h=z,
                             input_pos=input_pos,
                             propagate_pos=propagate_pos,)

            loss = 0

            # pde loss
            pde_loss_type = train_params['pde_loss_type'] if 'pde_loss_type' in train_params.keys() else None
            pde_loss_from_epoch = train_params['pde_loss_from_epoch'] if 'pde_loss_from_epoch' in train_params.keys() else 0
            if train_params['is_pde_loss']:
                if epoch >= pde_loss_from_epoch:
                    # FIXME: add PDE loss
                    loss_pde = loss_darcy_flow(u=output,
                                               a=label_a,
                                               force=dataset_params['beta'],
                                               batchsize=bs,
                                               resolution=128 // dataset_params['reduced_resolution'],
                                               alpha=train_params['pde_loss_alpha'],
                                               beta=train_params['pde_loss_beta'],
                                               type=pde_loss_type)

                    loss += train_params['pde_loss_weight'] * loss_pde
                    epoch_loss_dict['Loss_Phys'].append(loss_pde.detach().item())
                else:
                    epoch_loss_dict['Loss_Phys'].append(0)

            # data loss
            if train_params['is_data_loss']:
                loss_data = loss_data_fn(output, label)
                loss += train_params['data_loss_weight'] * loss_data
                epoch_loss_dict['Loss_Data'].append(loss_data.detach().item())

            # total loss
            epoch_loss_dict['Loss_Total'].append(loss.detach().item())

            if dataset_params['is_normalise']:
                output = label_normalizer.decode(output)
                label = label_normalizer.decode(label)

            # import torch.nn.functional as F
            # output = rearrange(output, 'b (h w) c -> b c h w', h=141)
            # output = output[..., 1:-1, 1:-1].contiguous()
            # output = F.pad(output, (1, 1, 1, 1), "constant", 0)
            # output = rearrange(output, 'b c h w -> b (h w) c')

            step_eval_dict = eval_2d(output, label, metrics_list=metrics_list)
            for metric_name in metrics_list:
                epoch_metrics_dict[metric_name].append(step_eval_dict[metric_name])

            # save array
            gt_array = label.cpu().numpy().reshape(bs, int(np.sqrt(label.shape[1])), int(np.sqrt(label.shape[1])), 1)  # (bs, h, w, 1)
            in_seq_array = input.cpu().numpy().reshape(bs, int(np.sqrt(input.shape[1])), int(np.sqrt(input.shape[1])), 3)  # (bs, h, w, 3)
            pred_array = output.cpu().numpy().reshape(bs, int(np.sqrt(output.shape[1])), int(np.sqrt(output.shape[1])), 1)  # (bs, h, w, 1)

            result_path = os.path.join(eval_dir)
            mkdir(result_path)
            np.savez(os.path.join(result_path, 'results_darcy_flow_vis.npz'),
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



def get_molifier(mesh, device):
    mollifier = 0.001 * torch.sin(np.pi * mesh[..., 0]) * torch.sin(np.pi * mesh[..., 1])
    return mollifier.to(device)


def loss_darcy_flow(u, a, force, batchsize, resolution, alpha=1, beta=1, type='FDMv3'):

    # u (bs, n_sample, 1)
    # a (bs, n_sample, 1)

    u = u.reshape(batchsize, resolution, resolution, 1)
    a = a.reshape(batchsize, resolution, resolution, 1)
    # loss_fn = LpLoss(size_average=True)
    loss_fn = nn.MSELoss()

    # pde term
    if type == 'FDM':
        Du = FDM_Darcy(u, a)
    elif type == 'FDMv2':
        Du = FDM_DarcyV2(u, a)
    elif type == 'FDMv3':
        Du = FDM_DarcyV3(u, a)
    else:
        raise NotImplementedError
    f = torch.ones(Du.shape, device=u.device) * force
    loss_phys = loss_fn(Du, f)

    # initial conditions
    loss_i = 0

    # boundary
    index_x = torch.cat([torch.tensor(range(0, resolution)), (resolution - 1) * torch.ones(resolution), torch.tensor(range(resolution-1, 1, -1)),
                         torch.zeros(resolution)], dim=0).long()
    index_y = torch.cat([(resolution - 1) * torch.ones(resolution), torch.tensor(range(resolution-1, 1, -1)), torch.zeros(resolution),
                         torch.tensor(range(0, resolution))], dim=0).long()
    # boundary conditions
    boundary_u = u[:, index_x, index_y]
    truth_u = torch.zeros(boundary_u.shape, device=u.device)
    loss_b = lploss.abs(boundary_u, truth_u)  # boundary condition loss

    loss = loss_phys + alpha * loss_b + beta * loss_i

    return loss


def FDM_Darcy(u, a, D=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    # ax = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    # ay = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    # uxx = (u[:, 2:, 1:-1] -2*u[:,1:-1,1:-1] +u[:, :-2, 1:-1]) / (dx**2)
    # uyy = (u[:, 1:-1, 2:] -2*u[:,1:-1,1:-1] +u[:, 1:-1, :-2]) / (dy**2)

    a = a[:, 1:-1, 1:-1]
    # u = u[:, 1:-1, 1:-1]
    # Du = -(ax*ux + ay*uy + a*uxx + a*uyy)

    # inner1 = torch.mean(a*(ux**2 + uy**2), dim=[1,2])
    # inner2 = torch.mean(f*u, dim=[1,2])
    # return 0.5*inner1 - inner2

    aux = a * ux
    auy = a * uy
    auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    Du = - (auxx + auyy)
    return Du



def FDM_DarcyV2(u, a, D=1):
    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    # ux: (batch, size-2, size-2)
    ux = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    uy = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)

    ax = (a[:, 2:, 1:-1] - a[:, :-2, 1:-1]) / (2 * dx)
    ay = (a[:, 1:-1, 2:] - a[:, 1:-1, :-2]) / (2 * dy)
    uxx = (u[:, 2:, 1:-1] -2*u[:,1:-1,1:-1] +u[:, :-2, 1:-1]) / (dx**2)
    uyy = (u[:, 1:-1, 2:] -2*u[:,1:-1,1:-1] +u[:, 1:-1, :-2]) / (dy**2)

    a = a[:, 1:-1, 1:-1]
    u = u[:, 1:-1, 1:-1]
    Du = -(ax*ux + ay*uy + a*uxx + a*uyy)

    # inner1 = torch.mean(a*(ux**2 + uy**2), dim=[1,2])
    # inner2 = torch.mean(f*u, dim=[1,2])
    # return 0.5*inner1 - inner2

    # aux = a * ux
    # auy = a * uy
    # auxx = (aux[:, 2:, 1:-1] - aux[:, :-2, 1:-1]) / (2 * dx)
    # auyy = (auy[:, 1:-1, 2:] - auy[:, 1:-1, :-2]) / (2 * dy)
    # Du = - (auxx + auyy)
    return Du


def FDM_DarcyV3(u, a, D=1):

    batchsize = u.size(0)
    size = u.size(1)
    u = u.reshape(batchsize, size, size)
    a = a.reshape(batchsize, size, size)
    dx = D / (size - 1)
    dy = dx

    grad_u = gradient_xy_scalar(u.unsqueeze(-1), dx, dy)
    grad_uy = grad_u[..., 0]
    grad_ux = grad_u[..., 1]
    a_grad_uy = a * grad_uy
    a_grad_ux = a * grad_ux

    grad_a_grad_uy = gradient_xy_scalar(a_grad_uy.unsqueeze(-1), dx, dy)[..., 0]
    grad_a_grad_ux = gradient_xy_scalar(a_grad_ux.unsqueeze(-1), dx, dy)[..., 1]

    Du = - (grad_a_grad_uy + grad_a_grad_ux)

    return Du
