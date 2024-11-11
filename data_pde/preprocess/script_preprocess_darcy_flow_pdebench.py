import os
from glob import glob
import torch
import numpy as np
import h5py
from shutil import rmtree

def mkdir(path=''):
    if not os.path.exists(path):
        os.makedirs(path)
        # print(f'create {path}')
    else:
        pass
        # print(f'{path} already exists.')


class H5Reader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(H5Reader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None

        self._load_file()

    def _load_file(self):
        self.data = h5py.File(self.file_path)

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field][()]

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# set path
# AYL
data_source_path = '/media/ssd/data_temp/PDE/data/DarcyFlow/PDEBench/'

# data name
# data_name = '2D_DarcyFlow_beta100.0_Train'
# data_name = '2D_DarcyFlow_beta10.0_Train'
data_name = '2D_DarcyFlow_beta1.0_Train'
# data_name = '2D_DarcyFlow_beta0.1_Train'
# data_name = '2D_DarcyFlow_beta0.01_Train'

data_source_path_raw = os.path.join(data_source_path, '{}.hdf5'.format(data_name))
data_folder_path_new = os.path.join(data_source_path, data_name)
mkdir(data_folder_path_new)

# load data list
data = H5Reader(data_source_path_raw)

nu = data.read_field('nu')
tensor = data.read_field('tensor')
x_co = data.read_field('x-coordinate')
y_co = data.read_field('y-coordinate')

num_all_file = tensor.shape[0]

# loop over file in each stage
for file_idx in range(num_all_file):

    # num_all_file=10000, 0000-0999: test, 1000-9999: train
    if file_idx < 1000:
        split_name = 'test'
    else:
        split_name = 'train'

    print('Processing {}/{}'.format(file_idx + 1, num_all_file))
    single_data = {
        'nu': nu[file_idx, ...],
        'tensor': tensor[file_idx, ...],
        'x-coordinate': x_co,
        'y-coordinate': y_co,
        }

    mkdir(os.path.join(data_folder_path_new, split_name))
    np.savez(os.path.join(data_folder_path_new, split_name, '2D_DarcyFlow_beta1.0_Train-{:04d}.npz'.format(file_idx)), **single_data)



