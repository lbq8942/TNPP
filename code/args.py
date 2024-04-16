import argparse
# hp_range = {#recommended range
#     #hyperparameters that are more important
#     "K": [1,2],  # 2
#     "num": [1,2, 3,4,5, 6,7, 8,9, 10 ]  # 5
#     # "num": [11,12,13,14,15]  # 5
# }#10

def load_args():
    parser = argparse.ArgumentParser('TPPCDF')

    parser.add_argument('--data', type=str,default="hawkes_ind", help='use which dataset')
    parser.add_argument('--bs', type=int,default=1)#目前普遍发现1会比较好。
    parser.add_argument('--hdim', type=int,default=32)#隐向量统一用一个。
    parser.add_argument('--layers',type=int, default=0)#要不要先对历史事件先相互注意一下，得到更加高级的信息。然后在用事件类型进行query,目前好像没有发现更好。
    parser.add_argument('--headsnum',type=int, default=4)
    parser.add_argument('--timeencode',type=int, default=0)#0:scalar,1:vector
    parser.add_argument('--norm',type=int, default=0)
    parser.add_argument('--gpu',type=int, default=-1)
    parser.add_argument('--regularization', type=float,default=1e-5)
    parser.add_argument('--lr',type=float, default=0.001)
    parser.add_argument('--max_epochs',type=int, default=1000)
    parser.add_argument('--display_step', type=int,default=1)
    parser.add_argument('--patience',type=int, default=10)
    parser.add_argument('--seed', type=int,default=0, help='random seed')
    parser.add_argument('--time_scale', type=float,default=1.0, help='time intervals in some datasets are very large')
    parser.add_argument('--seqlen', type=int,default=10, help='time intervals in some datasets are very large')
    #查看前面多少的

    parser.add_argument('--local', action="store_true", help='use local machine or remote machine')
    parser.add_argument('--testing', action="store_true", help='training or testing')#这个布尔值非常魔幻，--training False不行，会被当成布尔值，从而是True。
    parser.add_argument('--load_path', type=str, default="1",help="the path of model  when training is false")#如果是在测试的时候，该导入哪一个模型。
    parser.add_argument('--grid_search', action="store_true",help="search hyperparameter from the hp_range")#如果是在测试的时候，该导入哪一个模型。

    args = parser.parse_args()

    if args.local:
        args.pro_path="D:\lbq\lang\pythoncode\pycharm project\TPPBASE\TNPP"
        args.data_path="D:\lbq\lang\pythoncode\pycharm project\TPPBASE\RMTPP"
    else:#remote server to run this code
        args.pro_path="/data/liubingqing/debug/TPPBASE/TNPP"
        args.data_path = "/data/liubingqing/debug/TPPBASE/RMTPP"
    return args


