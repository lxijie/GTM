from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader,Dataset_weather,\
    Dataset_traffic,Dataset_elc
# from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from data_provider.utsdataset import UTSDataset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'utsd': UTSDataset,
    'weather':Dataset_weather,
    'traffic':Dataset_traffic,
    'elc':Dataset_elc,
    'msl':MSLSegLoader,
    'smap':SMAPSegLoader,
    'swat':SWATSegLoader,
    'psm':PSMSegLoader,
    'smd':SMDSegLoader
}


def data_provider(args, flag, data=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    # elif flag == 'pred':
    #     shuffle_flag = False
    #     drop_last = False
    #     batch_size = 1
    #     freq = args.freq
    #     Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    if args.data == 'utsd':
        data_set = Data(data, root_path=args.root_path, subset_name=r'UTSD-12G', input_len=args.seq_len, output_len=args.pred_len, flag=flag)
    elif args.task_name == 'anomaly_detection':
        data_set = Data(
            args,
            root_path=args.root_path,
            win_size = args.seq_len,
            flag = flag
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )   
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
