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

from utils.util_mesh import RandomMeshGenerator, SquareMeshGenerator


class DarcyFlowDataset(Dataset):
    """
    Dataset: Darcy Flow Dataset (PDE Bench)
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
        self.reduced_batch = dataset_params['reduced_batch']

        # task specific parameters
        self.beta = dataset_params['beta']

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
        self.label_y['a'] = []

        self._load_sample(self.data_paths[0])

        grid = self.input_x['feat'][0][..., -2:]
        grid_query = self.input_y['feat'][0]

        pwd = sklearn.metrics.pairwise_distances(grid)  # pwd: pair-wise distance: Arr(N_i, N_j). Arr_ij: the distance between ith & jth point.
        edge_index = np.vstack(np.where(pwd <= self.radius))  # Index_A & Index_B of the data on the condition
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        g = dgl.graph(([], []))
        g.add_edges(edge_index[0], edge_index[1])
        self.g = g

    def _prepare(self):

        # for one sample
        a = self.a
        u = self.u

        # set the value & key position (encoder)
        a_sampled = a.clone()
        u_sampled = u[0, ...]

        if self.mesh_type == 'random':

            a_sampled = rearrange(a_sampled, 'x y -> (x y) 1')
            u_sampled = rearrange(u_sampled, 'x y -> (x y) 1')

            # random sample
            sample_index_list = self.meshgenerator.sample()
            a_sampled = a_sampled[sample_index_list, ...]
            u_sampled = u_sampled[sample_index_list, ...]

        elif self.mesh_type == 'standard':
            # undersample on time and space
            a_sampled = a_sampled[::self.reduced_resolution, ::self.reduced_resolution]
            u_sampled = u_sampled[::self.reduced_resolution, ::self.reduced_resolution]

            a_sampled = rearrange(a_sampled, 'x y -> (x y) 1')
            u_sampled = rearrange(u_sampled, 'x y -> (x y) 1')

        else:
            raise NotImplementedError

        grid = self.meshgenerator.get_grid()  # (n_sample, 2)
        feat = torch.cat((a_sampled, grid), dim=1)

        # add to the list
        self.input_x['feat'].append(feat)

        # set the query position (additional encoder)
        a_query_sampled = a.clone()
        u_query_sampled = u[0, ...]

        if self.mesh_query_type in ['random', 'random_same']:

            a_query_sampled = rearrange(a_query_sampled, 'x y -> (x y) 1')
            u_query_sampled = rearrange(u_query_sampled, 'x y -> (x y) 1')

            # random sample
            sample_query_index_list = self.meshgenerator_query.sample()
            a_query_sampled = a_query_sampled[sample_query_index_list, ...]
            u_query_sampled = u_query_sampled[sample_query_index_list, ...]

        elif self.mesh_query_type == 'standard_same':
            # undersample on time and space
            a_query_sampled = a_query_sampled[::self.reduced_resolution, ::self.reduced_resolution]
            u_query_sampled = u_query_sampled[::self.reduced_resolution, ::self.reduced_resolution]

            a_query_sampled = rearrange(a_query_sampled, 'x y -> (x y) 1')
            u_query_sampled = rearrange(u_query_sampled, 'x y -> (x y) 1')

        elif self.mesh_query_type == 'standard_diag':
            # undersample on time and space
            a_query_sampled = a_query_sampled[(self.reduced_resolution-1)::self.reduced_resolution, (self.reduced_resolution-1)::self.reduced_resolution]
            u_query_sampled = u_query_sampled[(self.reduced_resolution-1)::self.reduced_resolution, (self.reduced_resolution-1)::self.reduced_resolution]

            a_query_sampled = rearrange(a_query_sampled, 'x y -> (x y) 1')
            u_query_sampled = rearrange(u_query_sampled, 'x y -> (x y) 1')

        elif self.mesh_query_type == 'standard_full_res':
            # undersample on time and space
            a_query_sampled = rearrange(a_query_sampled, 'x y -> (x y) 1')
            u_query_sampled = rearrange(u_query_sampled, 'x y -> (x y) 1')

        else:
            raise NotImplementedError

        grid_query = self.meshgenerator_query.get_grid()  # (n_sample, 2)
        self.input_y['feat'].append(grid_query)

        self.label_y['u'].append(u_query_sampled)
        self.label_y['a'].append(a_query_sampled)

    def _load_sample(self, data_path):

        # load data
        # Keys: ['nu', 'tensor', 'x-coordinate', 'y-coordinate']
        with np.load(data_path) as f:
            self.a = torch.from_numpy(f['nu'])  # (res_x, res_y)
            self.u = torch.from_numpy(f['tensor'])  # (1, res_x, res_y)
            self.x_co = torch.from_numpy(f['x-coordinate'])  # (res_x,)
            self.y_co = torch.from_numpy(f['y-coordinate'])  # (res_y,)

        self.real_space = [[self.x_co[0], self.x_co[-1]],
                           [self.y_co[0], self.y_co[-1]]]

        _, data_res_x, data_res_y, = self.u.shape  # (res_x, res_y, 2)
        assert data_res_x == data_res_y
        assert _ == 1
        self.res_full = data_res_x
        self.mesh_size = [self.res_full, self.res_full]
        self.res_grid = self.res_full // self.reduced_resolution

        # generate mesh
        if self.mesh_type == 'random':
            self.meshgenerator = RandomMeshGenerator(real_space=self.real_space,
                                                     mesh_size=self.mesh_size,
                                                     sample_size=self.random_mesh_sample_size,
                                                     downsample_rate=self.reduced_resolution,
                                                     is_diag=False)
        elif self.mesh_type == 'standard':
            self.meshgenerator = SquareMeshGenerator(real_space=self.real_space,
                                                     mesh_size=self.mesh_size,
                                                     downsample_rate=self.reduced_resolution,
                                                     is_diag=False)
        else:
            raise NotImplementedError

        # generate query mesh
        if self.mesh_query_type == 'random':
            self.meshgenerator_query = RandomMeshGenerator(real_space=self.real_space,
                                                           mesh_size=self.mesh_size,
                                                           sample_size=self.random_mesh_query_sample_size,
                                                           downsample_rate=self.reduced_resolution,
                                                           is_diag=False)
        elif self.mesh_query_type == 'random_same':
            assert self.mesh_type == 'random', 'mesh_type and mesh_query_type should be the same'
            self.meshgenerator_query = self.meshgenerator
        elif self.mesh_query_type == 'standard_same':
            self.meshgenerator_query = SquareMeshGenerator(real_space=self.real_space,
                                                           mesh_size=self.mesh_size,
                                                           downsample_rate=self.reduced_resolution,
                                                           is_diag=False)
        elif self.mesh_query_type == 'standard_diag':
            self.meshgenerator_query = SquareMeshGenerator(real_space=self.real_space,
                                                           mesh_size=self.mesh_size,
                                                           downsample_rate=self.reduced_resolution,
                                                           is_diag=True)
        elif self.mesh_query_type == 'standard_full_res':
            self.meshgenerator_query = SquareMeshGenerator(real_space=self.real_space,
                                                           mesh_size=self.mesh_size,
                                                           downsample_rate=1,
                                                           is_diag=False)
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
        self.label_y = {}
        self.label_y['u'] = []
        self.label_y['a'] = []

        self._load_sample(data_path)

        return self.input_x, self.input_y, self.label_y


class LoadDarcyFlowDatasetDGL(torch.utils.data.Dataset):

    def __init__(self, dataset_params):

        self.dataset_params = dataset_params
        self.name = dataset_params["dataset"]
        if 'train' in dataset_params['dataset_split']:
            self.train = DarcyFlowDataset(dataset_params, split='train')
        if 'val' in dataset_params['dataset_split']:
            self.val = DarcyFlowDataset(dataset_params, split='val')
        if 'test' in dataset_params['dataset_split']:
            self.test = DarcyFlowDataset(dataset_params, split='test')

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
