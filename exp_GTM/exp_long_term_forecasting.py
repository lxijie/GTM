from GTM.data_provider.data_factory import data_provider
from GTM.utils.tools import EarlyStopping, adjust_learning_rate, visual
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
os.environ['PYTHONWARNINGS'] = 'ignore'
import time
import warnings
import numpy as np
from GTM.data_provider.utsdataset import prepareUTSD
import deepspeed
import json
import matplotlib.pyplot as plt
from GTM.models import GTM
from collections import OrderedDict
# from utils.augmentation import run_augmentation,run_augmentation_single
os.environ['MASTER_PORT'] = '29501'

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast():
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'GTM': GTM,
        }
        self.model =self._build_model()
        if args.data == 'utsd':
            self.data = prepareUTSD(root_path='/data/dataset/train', subset_name=r'UTSD-12G', flag='train',
                                    input_len=self.args.seq_len, output_len=self.args.pred_len).load_data()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self._select_optimizer()
        with open('ds_config.json') as f:
            ds_config = json.load(f)
        self.model_parallel, self.optimizer, _, _ = deepspeed.initialize(args=self.args,model=self.model, optimizer=self.optimizer,model_parameters=parameters,
                                                                        config=ds_config,distributed_port=29502)
        if self.args.data != 'utsd':
            model_parameters = torch.load(self.args.pretrain_model_path)
            new_state_dict = OrderedDict()
            # 遍历当前的模型参数并修改键
            for key, value in model_parameters.items():
                # 去掉参数名称前的所有 'module.module.'
                # if 'head.weight' in new_state_dict:
                #     del new_state_dict['head.weight']
                # if 'head.bias' in new_state_dict:
                #     del new_state_dict['head.bias']
                new_key = 'module.'+key
                new_state_dict[new_key] = value.to(self.args.device)
                # new_state_dict[key] = value.to(self.args.device)
            self.model_parallel.load_state_dict(new_state_dict, strict=True)


    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.args.device)

    def _get_data(self, flag):
        if self.args.data =='utsd':
            data_set, data_loader = data_provider(self.args, flag, self.data)
        else:
            data_set, data_loader = data_provider(self.args,flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,time_gra) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.model_parallel.device)
                batch_y = batch_y.float().to(self.model_parallel.device)
                # encoder - decoder
                outputs = self.model(batch_x,time_gra)
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model_parallel(batch_x,time_gra)
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs
                    batch_y = patch_x
                else:
                    # f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs
                    batch_y = batch_y.to(self.model_parallel.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        criterion = self._select_criterion()
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, time_gra) in enumerate(train_loader):
                time_now = time.time()
                iter_count += 1
                self.optimizer.zero_grad()
                batch_x = batch_x.float().to(self.model_parallel.device)
                batch_y = batch_y.float().to(self.model_parallel.device)
                # decoder input
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model_parallel(batch_x,time_gra)
                    outputs = outputs
                    batch_y = patch_x
                else:
                    outputs = self.model_parallel(batch_x,time_gra)
                    outputs = outputs
                    batch_y = batch_y.to(self.model_parallel.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                if (i + 1) % 300 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    # time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.model_parallel.backward(loss)
                    self.model_parallel.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading models')
            model_parameters = torch.load('')

            new_state_dict = OrderedDict()
        #     # 遍历当前的模型参数并修改键
            for key, value in model_parameters.items():
                # 所有参数名称前加上 'module.module.'
                new_key = 'module.'+key
                new_state_dict[new_key] = value.to(self.args.device)
            self.model_parallel.load_state_dict(new_state_dict)

        lb_windows = []
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,time_gra) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.model_parallel.device)
                batch_y = batch_y.float().to(self.model_parallel.device)
                # encoder - decoder
                if 'pre_train' in self.args.task_name:
                    outputs, patch_x = self.model_parallel(batch_x,time_gra)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = patch_x
                else:
                    outputs = self.model_parallel(batch_x, time_gra)
                    outputs = outputs.detach().cpu().numpy()
                y = batch_y.detach().cpu().numpy()
                if i == 0:
                    lb_windows = batch_x.detach().cpu().numpy()
                    preds = outputs
                    trues = y
                else:
                    lb_window = batch_x.detach().cpu().numpy()
                    lb_windows = np.concatenate((lb_windows,lb_window),axis=0)
                    preds = np.concatenate((preds, outputs), axis=0)
                    trues = np.concatenate((trues, y), axis=0)
        print('test shape:', preds.shape, trues.shape)
        mse = F.mse_loss(torch.from_numpy(preds), torch.from_numpy(trues))
        mae = F.l1_loss(torch.from_numpy(preds), torch.from_numpy(trues))
        print('test shape:', preds.shape, trues.shape)
        print(f'the test loss {mse, mae}')
        return
