import datasets
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import os
os.environ["HF_DATASETS_PROGRESS_BARS"] = "0"

def convert_freq(freq):
    temp = freq
    freq = freq.resolution_string
    # 动态判断频率的合适单位
    if freq == 'ms' or 'us' or 'ns':
        gra = round(temp.value / 1000 / 1000)
        freq = 'ms'
    elif freq.lower() == 'D':
        gra = temp.days
    elif freq.lower() == 'h':
        gra = round(temp.value /1000 / 1000 / 1000 / 60 / 60)
    elif freq.lower() == 'min': # min
        gra = round(temp.value /1000 / 1000 / 1000 / 60)
    elif freq.lower() == 's':
        gra = round(temp.value /1000 / 1000 / 1000)

    # 自动进位
    while (freq == 'ms' and gra > 999):
        gra = round(gra / 1000)  # 进位到秒
        freq = 's'

    while (freq == 's' and gra > 999):
        gra = round(gra / 60)  # 进位到分钟
        freq = 't'

    while (freq == 't' and gra >= 60):
        gra = round(gra / 60)  # 进位到小时
        freq = 'h'

    while (freq == 'h' and gra >= 24):
        gra = round(gra / 24)  # 进位到天
        freq = 'd'

    return freq, gra

def get_time_gra(freq):
    freq, gra = convert_freq(freq)
    if freq.lower() == 'ms':
        return 'ms', [gra, 0,0,0,0]
    elif freq.lower() == 's':
        return 's', [0,gra,0,0,0]
    elif freq.lower() == 't':
        return 't',[0,0,gra,0,0]
    elif freq.lower() == 'h':
        return 'h', [0,0,0,gra,0]
    elif freq.lower() == 'd':
        return 'd', [0,0,0,0,gra]
    
class prepareUTSD():
    def __init__(self, root_path='/data/dataset/train', subset_name=r'UTSD-12G', flag='train', split=0.9,
                 input_len=None, output_len=None, scale=True, stride=192):
        self.root_path = root_path
        self.input_len = input_len
        self.output_len = output_len
        self.seq_len = input_len
        assert flag in ['train', 'val', 'test']
        assert split >= 0 and split <=1.0
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.scale = scale
        self.split = [0.7, 0.1, 0.2]
        self.stride = stride

        self.data_list = []
        self.freq_list = []
        self.time_gra = []
        self.n_window_list = []
        self.time_freq_dict = {
            "SelfRegulationSCP":[4,0,0,0,0],
            "AtrialFibrillation":[8,0,0,0,0],
            "IEEEPPG":[8,0,0,0,0],
            "TDBrain":[2,0,0,0,0],
            "CMIP6 ":[0,0,0,6,0]
        }
        self.subset_name = subset_name
    def load_data(self):
        dataset = datasets.load_from_disk(self.root_path)
        print('Indexing dataset...')
        train_set, val_set, test_set = [], [], []
        train_n_window_list, val_n_window_list, test_n_window_list = [], [], []
        # dataset = dataset.select(range(90000,90100))
        for item in dataset: # 数据集中已经分离多变量
            self.scaler = StandardScaler()
            freq = item['freq'] # 当前数据的freq
            data = item['target'] # 当前数据
            start = item['start']
            end = item['end']
            data = np.array(data).reshape(-1, 1)
            num_train = int(len(data) * self.split[0])
            num_val = int(len(data) * self.split[1])
            num_test = int(len(data) * self.split[2])
            border1s = [0, num_train - self.seq_len, num_train + num_val-self.seq_len]
            border2s = [num_train, num_train + num_val, len(data)]

            if self.scale:
                train_data = data[border1s[0]:border2s[0]] # 0: 806
                self.scaler.fit(train_data)
                data = self.scaler.transform(data)

            train_data = data[border1s[0]:border2s[0]]
            val_data = data[border1s[1]:border2s[1]]
            test_data = data[border1s[2]:border2s[2]]

            def process_data(data, item_id,start, end, freq, split_name, n_window_list):
                n_window = (len(data) - self.seq_len - self.output_len) // self.stride + 1
                if n_window < 1: # 不能生成完整序列，跳过数据项
                    return None, n_window_list

                # 累积窗口数量
                if len(n_window_list) == 0:
                    n_window_list.append(n_window)
                else:
                    n_window_list.append(n_window_list[-1] + n_window)

                if start == '' and end == '' and freq == '':
                    # self.time_gra.append('null')
                    gra = [0,0,0,0,0]
                else:
                    start_pd = pd.to_datetime(start)
                    end_pd = pd.to_datetime(end)
                    freq = (end_pd - start_pd) / (len(item['target']) - 1)
                    freq, gra = get_time_gra(freq)
                    # self.time_gra.append(gra)
                if freq == '':
                    freq = 'null'
                for key,value in self.time_freq_dict.items():
                    if key.lower() in item_id.lower():
                        gra = self.time_freq_dict[key]
                return {
                    'data_list': data,
                    'freq_list': freq,
                    'n_window_list': n_window_list[-1],
                    'time_gra': gra
                }, n_window_list
                # self.freq_list.append(freq)
                # self.data_list.append(data)
                # self.n_window_list.append(n_window if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_window) # 累积窗口数量
            train_item, train_n_window_list = process_data(data[border1s[0]:border2s[0]],item['item_id'], start, end, freq, 'train', train_n_window_list)
            val_item, val_n_window_list = process_data(data[border1s[1]:border2s[1]], item['item_id'],start, end, freq, 'val', val_n_window_list)
            test_item, test_n_window_list = process_data(data[border1s[2]:border2s[2]],item['item_id'], start, end, freq, 'test', test_n_window_list)

            if train_item:
                train_set.append(train_item)
            if val_item:
                val_set.append(val_item)
            if test_item:
                test_set.append(test_item)

            # 将 train_n_window_list 添加到 train_set 中
        train_set_dict = {
            'data_list': [item['data_list'] for item in train_set],
            'freq_list': [item['freq_list'] for item in train_set],
            'n_window_list': train_n_window_list,  # 添加整个窗口列表
            'time_gra': [item['time_gra'] for item in train_set]
        }

        val_set_dict = {
            'data_list': [item['data_list'] for item in val_set],
            'freq_list': [item['freq_list'] for item in val_set],
            'n_window_list': val_n_window_list,
            'time_gra': [item['time_gra'] for item in val_set]
        }

        test_set_dict = {
            'data_list': [item['data_list'] for item in test_set],
            'freq_list': [item['freq_list'] for item in test_set],
            'n_window_list': test_n_window_list,
            'time_gra': [item['time_gra'] for item in test_set]
        }

        return {'train': train_set_dict, 'val': val_set_dict, 'test': test_set_dict}
        # train_set = dict({'data_list':self.data_list[border1s[0]:border2s[0]], 'freq_list':self.freq_list[border1s[0]:border2s[0]], 'n_window_list': self.n_window_list[border1s[0]:border2s[0]], 'time_gra': self.time_gra[border1s[0]:border2s[0]]})
        # val_set = dict({'data_list':self.data_list[border1s[1]:border2s[1]], 'freq_list':self.freq_list[border1s[1]:border2s[1]], 'n_window_list': self.n_window_list[border1s[1]:border2s[1]], 'time_gra': self.time_gra[border1s[1]:border2s[1]]})
        # test_set = dict({'data_list':self.data_list[border1s[2]:border2s[2]], 'freq_list':self.freq_list[border1s[2]:border2s[2]], 'n_window_list': self.n_window_list[border1s[2]:border2s[2]], 'time_gra': self.time_gra[border1s[2]:border2s[2]]})
        # return [train_set, val_set, test_set]


"""
All single-variate series in UTSD are divided into (input-output) windows with a uniform length based on S3.
"""
class UTSDataset(Dataset):
    def __init__(self, data, root_path='/data/dataset/train', subset_name=r'UTSD-12G', flag='train', split=0.9,
                 input_len=None, output_len=None, scale=True, stride=192):
        self.root_path = root_path
        self.input_len = input_len
        self.output_len = output_len
        self.seq_len = input_len
        assert flag in ['train', 'val', 'test']
        assert split >= 0 and split <=1.0
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.scale = scale
        self.split = [0.7, 0.1, 0.2]
        self.stride = stride

        self.data_list = []
        self.freq_list = []
        self.time_gra = []
        self.n_window_list = []

        self.flag = flag
        self.subset_name = subset_name
        self.__read_data__(data)

    def __read_data__(self, data):
        data = data[self.flag]
        self.data_list = data['data_list']
        self.freq_list = data['freq_list']
        self.time_gra = data['time_gra']
        self.n_window_list = data['n_window_list']
        # dataset = dataset
        # dataset = datasets.load_from_disk(self.root_path)
        # # # dataset = datasets.load_dataset("thuml/UTSD", self.subset_name, split='train')
        # # # split='train' contains all the time series, which have not been divided into splits, 
        # # # you can split them by yourself, or use our default split as train:val = 9:1
        # print('Indexing dataset...')
        # for item in tqdm(dataset): # 数据集中已经分离多变量
        #     self.scaler = StandardScaler()
        #     freq = item['freq'] # 当前数据的freq
        #     data = item['target'] # 当前数据
        #     start = item['start']
        #     end = item['end']
        #     data = np.array(data).reshape(-1, 1)
        #     num_train = int(len(data) * self.split[0])
        #     num_val = int(len(data) * self.split[1])
        #     num_test = int(len(data) * self.split[2])
        #     border1s = [0, num_train - self.seq_len, num_train + num_val-self.seq_len]
        #     border2s = [num_train, num_train + num_val, len(data)]

        #     border1 = border1s[self.set_type]
        #     border2 = border2s[self.set_type]

        #     if self.scale:
        #         train_data = data[border1s[0]:border2s[0]] # 0: 806
        #         self.scaler.fit(train_data)
        #         data = self.scaler.transform(data)

        #     data = data[border1:border2]

        #     n_window = (len(data) - self.seq_len - self.output_len) // self.stride + 1
        #     if n_window < 1: # 不能生成完整序列，跳过数据项
        #         continue

        #     if start == '' and end == '' and freq == '':
        #         self.time_gra.append('null')
        #     else:
        #         # if freq == '':
        #         start_pd = pd.to_datetime(start)
        #         end_pd = pd.to_datetime(end)
        #         freq = (end_pd - start_pd) / (len(item['target']) - 1)
        #         freq, gra = get_time_gra(freq)
        #         self.time_gra.append(gra)
        #         # else:
        #         #     # 生成时间序列
        #         #     time_series = pd.date_range(start=start, end=end, freq=freq)
        #         #     # print(time_series)

        #         #     # 计算时间间隔（粒度）
        #         #     freq = time_series[1] - time_series[0]
        #         #     # print(f"时间粒度: {freq}")
        #         #     freq, gra = get_time_gra(freq)
        #         #     self.time_gra.append(gra)

        #     # if len(self.time_gra)==18260:
        #     #     print()
        #     if freq == '':
        #         freq = 'null'
        #     self.freq_list.append(freq)
        #     self.data_list.append(data)
        #     self.n_window_list.append(n_window if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_window) # 累积窗口数量


    def __getitem__(self, index):
        # you can wirte your own processing code here
        # 这里的dataset_index指list中的索引而不是真实数据集
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]: # 累计窗口大小进行比较，以确定 index 属于哪个子数据集
            dataset_index += 1

        index = index - self.n_window_list[dataset_index - 1] if dataset_index > 0 else index
        n_timepoint = (len(self.data_list[dataset_index]) - self.seq_len - self.output_len) // self.stride + 1 # 为了确定开始

        s_begin = index % n_timepoint
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        p_begin = s_end
        p_end = p_begin + self.output_len
        seq_x = self.data_list[dataset_index][s_begin:s_end, :]
        seq_y = self.data_list[dataset_index][p_begin:p_end, :]
        freq = self.freq_list[dataset_index]
        gra = self.time_gra[dataset_index]

        if seq_x.size != self.seq_len or seq_y.size != self.output_len:
            print(f"Unexpected size at index {index}: seq_x size = {seq_x.size}, seq_y size = {seq_y.size}")

        # 时间粒度信息[ms,s,min,hour,day]
        # gra = [0,1,0,0,0]
        return seq_x, seq_y, gra

    def __len__(self):
        # return len(self.data_list)
        return self.n_window_list[-1]


# See ```download_dataset.py``` to download the dataset first
# if __name__ == '__main__':
#     # dataset = UTSDataset(subset_name=r'UTSD-1G', input_len=672, output_len=0, flag='train')
#     dataset = UTSDataset(subset_name=r'UTSD-12G', input_len=720, output_len=96, flag='train')
#     dataloader = DataLoader(dataset, batch_size=2048, drop_last=True)
#     for i, (seq_x, seq_y, frep, gra) in enumerate(dataloader):
#         print(f"Batch {i}: seq_x size: {seq_x.size()}, seq_y size: {seq_y.size()}, frep: {len(frep)}")
#     print(f'total {len(dataset)} time series windows (sentence)')
