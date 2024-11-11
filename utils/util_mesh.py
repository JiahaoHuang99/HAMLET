import scipy.io
import h5py
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
from torch_geometric.data import Data
from scipy.ndimage import gaussian_filter


class SquareMeshGenerator(object):
    def __init__(self, real_space, mesh_size, downsample_rate=1, is_diag=False, add_one=False):
        super(SquareMeshGenerator, self).__init__()

        self.d = len(real_space)  # d dimension
        self.s = mesh_size[0]

        assert len(mesh_size) == self.d

        if self.d == 1:
            raise NotImplementedError('Bug here!')
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))

            # downsample
            self.n = self.n // downsample_rate
            if is_diag:
                self.grid = self.grid[(downsample_rate - 1)::downsample_rate, :]
            else:
                self.grid = self.grid[::downsample_rate, :]
            self.grid = self.grid.reshape((self.n, 1))

        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                # self.n *= mesh_size[j]
                # downsample
                if add_one:
                    self.n *= (mesh_size[j] - 1) // downsample_rate + 1
                else:
                    self.n *= mesh_size[j] // downsample_rate

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

            self.grid = self.grid.reshape((self.s, self.s, 2))
            if is_diag:
                self.grid = self.grid[(downsample_rate - 1)::downsample_rate, (downsample_rate - 1)::downsample_rate, :]
            else:
                self.grid = self.grid[::downsample_rate, ::downsample_rate, :]
            self.grid = self.grid.reshape((self.n, 2))

    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    def get_node_num(self):
        return self.n

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid)  # pwd: pair-wise distance: Arr(N_i, N_j). Arr_ij: the distance between ith & jth point.
        self.edge_index = np.vstack(np.where(pwd <= r))  # Index_A & Index_B of the data on the condition
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    # FIXME: Please check! Not clear
    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)


class RandomMeshGenerator(object):
    def __init__(self, real_space, mesh_size, sample_size, downsample_rate=1, is_diag=False):
        super(RandomMeshGenerator, self).__init__()

        self.d = len(real_space)  # d dimension
        self.s = mesh_size[0]
        self.m = sample_size

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))

            # downsample
            self.n = self.n // downsample_rate
            if is_diag:
                self.grid = self.grid[(downsample_rate - 1)::downsample_rate, :]
            else:
                self.grid = self.grid[::downsample_rate, :]
            self.grid = self.grid.reshape((self.n, 1))

        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]
            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

            # downsample
            self.n = self.n // downsample_rate // downsample_rate
            self.grid = self.grid.reshape((self.s, self.s, 2))
            if is_diag:
                self.grid = self.grid[(downsample_rate - 1)::downsample_rate, (downsample_rate - 1)::downsample_rate, :]
            else:
                self.grid = self.grid[::downsample_rate, ::downsample_rate, :]
            self.grid = self.grid.reshape((self.n, 2))

        if self.m > self.n:
            self.m = self.n

        self.idx = np.array(range(self.n))
        self.grid_sample = self.grid

    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[:self.m]
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)

    def get_node_num(self):
        return self.m

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    # FIXME: Please check! Not clear
    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                theta = theta[self.idx]
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                theta = theta[self.idx]
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)


class CustomCoodGenerator(object):
    def __init__(self, grid):
        super(CustomCoodGenerator, self).__init__()

        self.grid = grid
        self.n = grid.shape[0]

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid)  # pwd: pair-wise distance: Arr(N_i, N_j). Arr_ij: the distance between ith & jth point.
        self.edge_index = np.vstack(np.where(pwd <= r))  # Index_A & Index_B of the data on the condition
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    def get_node_num(self):
        return self.n

    # FIXME: Please check! Not clear
    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)


class RandomCustomCoodGenerator(object):
    def __init__(self, grid, sample_size):
        super(RandomCustomCoodGenerator, self).__init__()

        self.grid = grid
        self.n = grid.shape[0]
        self.m = sample_size

        if self.m > self.n:
            self.m = self.n

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)  # pwd: pair-wise distance: Arr(N_i, N_j). Arr_ij: the distance between ith & jth point.
        self.edge_index = np.vstack(np.where(pwd <= r))  # Index_A & Index_B of the data on the condition
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[:self.m]
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)

    def get_node_num(self):
        return self.m

    # FIXME: Please check! Not clear
    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

