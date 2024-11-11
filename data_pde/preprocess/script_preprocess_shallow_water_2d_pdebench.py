import os
from glob import glob
import torch
import numpy as np
import h5py
from shutil import rmtree
from matplotlib import pyplot as plt

def mkdir(path=''):
    if not os.path.exists(path):
        os.makedirs(path)
        # print(f'create {path}')
    else:
        pass
        # print(f'{path} already exists.')


if __name__ == '__main__':

    # number of data
    n = 1000

    # set path

    # AYL
    data_source_path = '/media/ssd/data_temp/PDE/data/ShallowWater/PDEBench'
    data_file_path_raw = os.path.join(data_source_path, 'raw/2D_rdb_NA_NA.h5')

    data_folder_path_new = os.path.join(data_source_path, 'npz')
    mkdir(data_folder_path_new)

    # read data
    print('Loading all data for mean and std...')
    all_data_collection = h5py.File(data_file_path_raw)
    data_temp_list = []
    for key in all_data_collection.keys():
        data_temp = all_data_collection[key]['data'][()]
        data_temp_list.append(data_temp)
    data_all = np.array(data_temp_list)
    data_mean, data_std = data_all.mean(), data_all.std()
    import gc
    del data_all
    del data_temp_list
    gc.collect()

    for i in range(n):
        data_name = '2D_rdb_NA_NA.h5'
        data_index = '{:04d}'.format(i)

        # get single data
        data = all_data_collection[data_index]

        # < KeysViewHDF5['data', 'grid'] >
        print('Processing {}/{}'.format(i + 1, 1000))
        batch_data = {
            'data': data['data'][()],
            'grid_x': data['grid']['x'][()],
            'grid_y': data['grid']['y'][()],
            'grid_t': data['grid']['t'][()],
            'data_name': data_name,
            'data_index': data_index,
            'data_mean': data_mean,
            'data_std': data_std,
        }

        mkdir(os.path.join(data_folder_path_new, 'all'))
        np.savez(os.path.join(data_folder_path_new, 'all', '{}-{}.npz'.format(data_name, data_index)), **batch_data)


    # # ---------------------------
    # # FIXME: DEBUG
    # save_dir = '/home/jh/physics_graph_transformer/physics_graph_transformer/tmp/hist/sw'
    # mkdir(save_dir)
    #
    # u_list = []
    # for i in range(n):
    #     u_list.append(all_data_collection['{:04d}'.format(i)]['data'][()])
    #
    # u_all = np.stack(u_list, axis=0)
    # t = u_all.shape[1]
    # for i in range(t):
    #     u_t = u_all[:, i, :, :, :].reshape(-1)
    #     # plt.imshow(u_t)
    #     plt.hist(u_t)
    #     mkdir(os.path.join(save_dir, 'u'))
    #     plt.savefig(os.path.join(save_dir, 'u', '{:03d}.png'.format(i)))
    #     plt.close()
    #     print('save fig {}'.format(i))
    #
    # # FIXME: DEBUG
    # # ---------------------------------