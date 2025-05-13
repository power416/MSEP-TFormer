import os
import torch
import torch.utils.data as data
import numpy as np
import random
from options import *

random.seed(1143)

class Dataset(data.Dataset):
    def __init__(self, path, file_name, norm_mode=opt.norm_mode):
        super(Dataset, self).__init__()
        self.norm_mode = norm_mode
        self.data_dir = os.path.join(path, file_name)
        self.data_list = np.load(self.data_dir, allow_pickle=True)

    def __getitem__(self, index):
        df = self.data_list[index]
        
        df_data = df['data']
        d = df_data.shape[0]

        label_m = np.zeros(1)
        label_e = np.zeros(1)
        label_d = np.zeros(1)
        label_t = np.zeros(1)
        
        label_p = np.zeros([1, d])
        label_s = np.zeros([1, d])

        mag = df['source_magnitude']
        epicenter = df['source_distance_km']
        depth = df['source_depth_km']
        travel = df['p_travel_sec']
        p = df['p_arrival_sample']
        s = df['s_arrival_sample']

        label_m[:] = float(mag)
        label_e[:] = float(epicenter)
        label_d[:] = float(depth)
        label_t[:] = float(travel)

        if self.norm_mode:
            df_data = self._normalize(df_data, self.norm_mode)

        if p and (p - 20 >= 0) and (p + 20 < d):
            # print(1)
            label_p[:, int(p - 20):int(p + 20)] = np.exp(
                -(np.arange(int(p - 20), int(p + 20)) - p) ** 2 / (2 * (10) ** 2))[:int(p + 20) - int(p - 20)]
        elif p and (p + 20 >= d):
            label_p[:, int(p - 20): d] = np.exp(
                -(np.arange(int(p - 20), int(p + 20)) - p) ** 2 / (2 * (10) ** 2))[:(d - int(p - 20))]
        elif p and (p - 20 < 0):
            label_p[:, 0: int(p + 20)] = np.exp(
                -(np.arange(int(p - 20), int(p + 20)) - p) ** 2 / (2 * (10) ** 2))[int(20 - p):]

        if s and (s - 20 >= 0) and (s + 20 < d):
            label_s[:, int(s - 20): int(s + 20)] = np.exp(
                -(np.arange(int(s - 20), int(s + 20)) - s) ** 2 / (2 * (10) ** 2))[:int(s + 20) - int(s - 20)]
        elif s and (s + 20 >= d):
            label_s[:, int(s - 20): d] = np.exp(
                -(np.arange(int(s - 20), int(s + 20)) - s) ** 2 / (2 * (10) ** 2))[:(d - int(s - 20))]
        elif s and (s - 20 < 0):
            label_s[:, 0: int(s + 20)] = np.exp(
                -(np.arange(int(s - 20), int(s + 20)) - s) ** 2 / (2 * (10) ** 2))[int(20 - s):]

        df_data, label_m, label_e, label_d, label_t, label_p, label_s = self.augData(df_data, label_m, label_e, label_d, label_t, label_p, label_s)

        return df_data, label_m, label_e, label_d, label_t, label_p, label_s

    def augData(self, x, x_m, x_e, x_d, x_t, x_p, x_s):

        x = torch.from_numpy(x).float()
        x = torch.permute(x, [1, 0])

        x_m = torch.from_numpy(x_m).float()
        x_e = torch.from_numpy(x_e).float()
        x_d = torch.from_numpy(x_d).float()
        x_t = torch.from_numpy(x_t).float()
        x_p = torch.from_numpy(x_p).float()
        x_s = torch.from_numpy(x_s).float()

        return x, x_m, x_e, x_d, x_t, x_p, x_s
    
    def _normalize(self, data, mode='std'):
        # Normalize waveforms in each batch
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            max_data[max_data == 0] = 1
            data /= max_data

        elif mode == 'std':
            std_data = np.std(data, axis=0, keepdims=True)
            std_data[std_data == 0] = 1
            data /= std_data
        return data
    
    def __len__(self):

        return len(self.data_list)
