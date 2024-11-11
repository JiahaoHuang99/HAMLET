import os
import dgl
import time
import numpy as np
import sklearn
import hashlib
import networkx as nx
from math import ceil
from scipy import sparse as sp
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from glob import glob
from einops import rearrange, repeat

from utils.util_mesh import CustomCoodGenerator, RandomCustomCoodGenerator


class ShallowWater2DDataset(Dataset):
    """
    Dataset: Shallow Water 2D Dataset
    Source: https://doi.org/10.18419/darus-2986
    This is a folder dataset (preprocessed).
    Folder:
    - Files (*.npz):
        - Sample (x1)

    """
    def __init__(self, dataset_params, split):
        # load configuration
        self.split = split  # train, val, test
        dataset_params = dataset_params[split]
        self.dataset_params = dataset_params

        # path to dataset folder
        data_folder_path = dataset_params['dataset_path']

        # dataset number and split configuration
        self.n_all_samples = dataset_params['n_all_samples']

        # resolution and domain range configuration
        self.reduced_resolution = dataset_params['reduced_resolution']
        self.reduced_resolution_t = dataset_params['reduced_resolution_t']
        self.reduced_batch = dataset_params['reduced_batch']

        # task specific parameters
        self.in_seq = dataset_params['in_seq']
        self.out_seq = dataset_params['out_seq']

        # graph parameters
        self.radius = dataset_params['radius']
        self.mesh_type = dataset_params['mesh_type']
        self.random_mesh_sample_size = dataset_params['random_mesh_sample_size']
        self.mesh_query_type = dataset_params['mesh_query_type']
        self.random_mesh_query_sample_size = dataset_params['random_mesh_query_sample_size']
        self.node_attr_chnl = dataset_params['node_attr_chnl']
        self.edge_attr_chnl = dataset_params['edge_attr_chnl']
        self.use_edge_attr = True if self.edge_attr_chnl != 0 else False

        # load data list
        self.data_paths_all = sorted(glob(os.path.join(data_folder_path, '*.npz')))
        if self.n_all_samples != len(self.data_paths_all):
            print("Warning: n_all_samples is not equal to the number of files in the folder")
        self.data_paths_all = self.data_paths_all[:self.n_all_samples]

        # dataset has been split into train, test (folder)
        self.n_samples = self.n_all_samples
        self.data_paths = self.data_paths_all

        # load an example
        self.input_x ={}
        self.input_x['feat'] = []
        self.input_y = {}
        self.input_y['feat'] = []
        self.label_y = {}
        self.label_y['u'] = []
        self.label_y['u_mean'] = []
        self.label_y['u_std'] = []

        self._load_sample(self.data_paths[0])

        grid = self.input_x['feat'][0][..., -2:]
        grid_query = self.input_y['feat'][0]

        pwd = sklearn.metrics.pairwise_distances(grid)  # pwd: pair-wise distance: Arr(N_i, N_j). Arr_ij: the distance between ith & jth point.
        edge_index = np.vstack(np.where(pwd <= self.radius))  # Index_A & Index_B of the data on the condition
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        g = dgl.graph(([], []))
        g.add_edges(edge_index[0], edge_index[1])
        self.g = g

        pwd_query = sklearn.metrics.pairwise_distances(grid_query)  # pwd: pair-wise distance: Arr(N_i, N_j). Arr_ij: the distance between ith & jth point.
        edge_index_query = np.vstack(np.where(pwd_query <= self.radius))  # Index_A & Index_B of the data on the condition
        edge_index_query = torch.tensor(edge_index_query, dtype=torch.long)
        g_query = dgl.graph(([], []))
        g_query.add_edges(edge_index_query[0], edge_index_query[1])
        self.g_query = g_query

    def _prepare(self):

        # for one sample
        t = self.grid_t
        u = self.u


        # set the value & key position (encoder)
        u_sampled = u.clone()

        if self.mesh_type == 'random_custom':

            u_sampled = rearrange(u_sampled, 't x y a -> x y t a')

            # undersample on time and space
            u_sampled = u_sampled[:, :, ::self.reduced_resolution_t, :]
            t_sampled = t[::self.reduced_resolution_t]

            u_sampled = rearrange(u_sampled, 'x y t a -> (x y) t a')

            # random sample
            sample_index_list = self.meshgenerator.sample()
            u_sampled = u_sampled[sample_index_list, ...]

        elif self.mesh_type == 'custom':

            u_sampled = rearrange(u_sampled, 't x y a -> x y t a')

            # undersample on time and space
            u_sampled = u_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
            t_sampled = t[::self.reduced_resolution_t]

            u_sampled = rearrange(u_sampled, 'x y t 1 -> (x y) t 1')

        else:
            raise NotImplementedError

        grid = self.meshgenerator.get_grid()  # (n_sample, 2)

        u_sampled_inseq = rearrange(u_sampled[:, :self.in_seq, :], 'n t a -> n (t a)')

        feat = torch.cat((u_sampled_inseq,
                          grid), dim=1)

        # add to the list
        self.input_x['feat'].append(feat)

        # set the query position (additional encoder)
        u_query_sampled = u.clone()

        if self.mesh_query_type == 'random_custom':

            u_query_sampled = rearrange(u_query_sampled, 't x y a -> x y t a')

            # undersample on time and space
            u_query_sampled = u_query_sampled[:, :, ::self.reduced_resolution_t, :]
            t_query_sampled = t[::self.reduced_resolution_t]

            u_query_sampled = rearrange(u_query_sampled, 'x y t 1 -> (x y) t 1')

            # random sample
            sample_query_index_list = self.meshgenerator_query.sample()
            u_query_sampled = u_query_sampled[sample_query_index_list, ...]

        elif self.mesh_query_type == 'custom':

            u_query_sampled = rearrange(u_query_sampled, 't x y a -> x y t a')

            # undersample on time and space
            u_query_sampled = u_query_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
            t_query_sampled = t[::self.reduced_resolution_t]

            u_query_sampled = rearrange(u_query_sampled, 'x y t 1 -> (x y) t 1')

        else:
            raise NotImplementedError

        grid_query = self.meshgenerator_query.get_grid()  # (n_sample, 2)
        self.input_y['feat'].append(grid_query)

        self.label_y['u'].append(u_query_sampled)
        self.label_y['u_mean'].append(self.u_mean)
        self.label_y['u_std'].append(self.u_std)

    def _load_sample(self, data_path):

        # load data
        with np.load(data_path) as f:

            # self.data = torch.from_numpy(f['data'][()].astype(np.float32))
            self.data = torch.from_numpy(f['data'][()])  # (res_t, res_x, res_y, 2)

            self.u = self.data[:, :, :, :]
            self.u_mean = torch.from_numpy(f['data_mean'].astype(np.float32))
            self.u_std = torch.from_numpy(f['data_std'].astype(np.float32))
            self.grid_t = torch.from_numpy(f['grid_t'][()])  # res_t = 101
            self.grid_x = torch.from_numpy(f['grid_x'][()])  # res_x = 128
            self.grid_y = torch.from_numpy(f['grid_y'][()])  # res_y = 128

        data_res_t, data_res_x, data_res_y, _ = self.data.shape  # (res_t, res_x, res_y, 1)
        assert data_res_x == data_res_y
        assert _ == 1
        self.res_full = data_res_x
        self.mesh_size = [self.res_full, self.res_full]
        self.res_grid = self.res_full // self.reduced_resolution
        self.data_res_t = data_res_t
        self.res_time = ceil(self.data_res_t / self.reduced_resolution_t)
        assert self.in_seq + self.out_seq == self.res_time

        # mesh grid
        self.grid_x_sampled = self.grid_x[::self.reduced_resolution]
        self.grid_y_sampled = self.grid_y[::self.reduced_resolution]
        gx, gy = torch.meshgrid(self.grid_x_sampled, self.grid_y_sampled)
        self.grid = torch.stack((gy, gx), dim=-1).reshape(-1, 2)  # (res_x * res_y, 2)

        # generate mesh
        if self.mesh_type == 'random_custom':
            self.meshgenerator = RandomCustomCoodGenerator(grid=self.grid.numpy(),
                                                           sample_size=self.random_mesh_sample_size)
        elif self.mesh_type == 'custom':
            self.meshgenerator = CustomCoodGenerator(grid=self.grid.numpy())
        else:
            raise NotImplementedError

        # mesh grid query
        self.grid_x_query_sampled = self.grid_x[::self.reduced_resolution]
        self.grid_y_query_sampled = self.grid_y[::self.reduced_resolution]
        gx, gy = torch.meshgrid(self.grid_x_query_sampled, self.grid_y_query_sampled)
        self.grid_query = torch.stack((gy, gx), dim=-1).reshape(-1, 2)  # (res_x * res_y, 2)

        # set the query position (additional encoder)
        if self.mesh_type == 'random_custom':
            self.meshgenerator_query = RandomCustomCoodGenerator(grid=self.grid_query.numpy(),
                                                                 sample_size=self.random_mesh_sample_size)
        elif self.mesh_type == 'custom':
            self.meshgenerator_query = CustomCoodGenerator(grid=self.grid_query.numpy())
        else:
            raise NotImplementedError

        self._prepare()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):

        data_path = self.data_paths[idx]

        self.input_x ={}
        self.input_x['feat'] = []
        self.input_x['graph'] = [self.g]
        self.input_y = {}
        self.input_y['feat'] = []
        self.input_x['graph'] = [self.g_query]
        self.label_y = {}
        self.label_y['u'] = []
        self.label_y['u_mean'] = []
        self.label_y['u_std'] = []

        self._load_sample(data_path)

        return self.input_x, self.input_y, self.label_y


class LoadShallowWater2DDatasetDGL(torch.utils.data.Dataset):

    def __init__(self, dataset_params):

        self.dataset_params = dataset_params
        self.name = dataset_params["dataset"]
        if 'train' in dataset_params['dataset_split']:
            self.train = ShallowWater2DDataset(dataset_params, split='train')
        if 'val' in dataset_params['dataset_split']:
            self.val = ShallowWater2DDataset(dataset_params, split='val')
        if 'test' in dataset_params['dataset_split']:
            self.test = ShallowWater2DDataset(dataset_params, split='test')

    def collate(self, samples):
        # form a mini batch from a given list of samples = [(graph, label) pairs]
        # The input samples is a list of pairs (graph, label).
        input_x, input_y, labels = map(list, zip(*samples))

        # convert label format
        new_input_x = reformat_dict(input_x)
        new_input_y = reformat_dict(input_y)
        new_labels = reformat_dict(labels)

        return new_input_x, new_input_y, new_labels



def reformat_dict(labels):
    new_labels = {}
    for idx, label_dict in enumerate(labels):
        for key in label_dict.keys():
            if idx == 0:
                new_labels[key] = []
            new_labels[key].append(label_dict[key][0])
    for key in new_labels.keys():
        new_labels[key] = torch.stack(new_labels[key], dim=0) if key != 'graph' else new_labels['graph'][0]
    return new_labels
