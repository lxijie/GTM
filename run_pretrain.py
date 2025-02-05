import sys
import deepspeed
import argparse
import os
import torch
from exp_GTM.exp_pre_train import Exp_Long_Term_Forecast
# from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
import random
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

os.environ['DEEPSPEED_CONFIG'] = '{"ports":{"master_port":29501}}'
if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='GTM')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='models id')
    parser.add_argument('--models', type=str, default='Autoformer',
                        help='models name, options: []')

    # data loader
    parser.add_argument('--train_len', type=float, default=0.7, help='train:vali:test')
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of models checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # models define
    parser.add_argument('--k_adaptive', type=bool, default=False, help='k_adaptive')
    parser.add_argument('--individual', type=bool, default=False, help='individual')
    parser.add_argument('--instance_normalization', type=bool, default=True, help='Instance Normalization')

    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of models')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_layers', type=int, default=12, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='SMAPE', help='loss function')
    parser.add_argument('--lradj', type=str, default='cosine', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.data = 'utsd'  # 'custom_new'#
    args.root_path = '/data/dataset/train'
    # basic config
    args.task_name = 'pre_train'  # [long_term_forecast,pre_train,anomaly_detection]
    args.logs_dir = './logs_new/pre_train/'
    args.model_id = 'GTM'

    args.pretrain_model_path = ''
    # training param
    args.model = 'GTM'
    args.train_epochs = 30
    args.patience = 5
    args.learning_rate = 1e-5
    args.weight_decay = 0
    args.lradj = 'cosine'
    args.num_gpus = 5
    # models param
    args.d_model = 768
    args.k_adaptive = False
    args.individual = False
    args.instance_normalization = True
    args.dropout = 0.1
    args.enc_in = 1
    args.dec_in = 1
    args.c_out = 1
    args.seq_len = 1440
    args.pred_len = 0
    args.d_layers = 12
    args.batch_size = 1024
    args.patch_len = 96
    args.stride = 96
    args.device = torch.device('cuda:{}'.format(args.gpu))
    print('Args in experiment:')
    print(args)

    args.is_training = True
    if args.is_training:
        Exp = Exp_Long_Term_Forecast(args)
        setting = '{}_{}_{}_{}_{}_ft{}_id{}_sl{}_pl{}_dm{}_df{}_lr{}_bs{}'.format(
            args.task_name,
            args.model_id,
            args.data,
            args.model,
            args.seasonal_patterns,
            args.features,
            args.individual,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.learning_rate,
            args.batch_size)
        # print('parameters：',sum(p.numel() for p in Exp.model.parameters()))
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        Exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        Exp.test(setting)
        torch.cuda.empty_cache()
    else:
        Exp = Exp_Long_Term_Forecast(args)
        setting = '{}_{}_{}_{}_{}_ft{}_id{}_sl{}_pl{}_dm{}_df{}_lr{}_bs{}'.format(
            args.task_name,
            args.model_id,
            args.data,
            args.model,
            args.seasonal_patterns,
            args.features,
            args.individual,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.learning_rate,
            args.batch_size)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        Exp.test(setting, test=1)
        torch.cuda.empty_cache()

