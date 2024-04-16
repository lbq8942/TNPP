import dpp
import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from dpp.metrics import aggregate_loss_over_dataloader, sampling_plot#这两个东西，我有点忘记了，干什么用的了。
from args import *
from log import *
from util import *
import random
# Config
def set_random_seed(seed=42):
    torch.manual_seed(seed)#torch的cpu随机性
    torch.cuda.manual_seed_all(seed)#torch的gpu随机性
    torch.backends.cudnn.benchmark = False#保证gpu每次都选择相同的算法，但是不保证该算法是deterministic的。
    torch.backends.cudnn.deterministic = True#紧接着上面，保证算法是deterministic的。
    np.random.seed(seed)#np的随机性。
    random.seed(seed)#python的随机性。
    os.environ['PYTHONHASHSEED'] = str(seed)#设置python哈希种子


# # Training config
# regularization = 1e-5  # L2 regularization parameter
# learning_rate = 1e-3   # Learning rate for Adam optimizer
# max_epochs = 1000      # For how many epochs to train
# display_step = 5       # Display training statistics after every display_step
# patience = 50          # After how many consecutive epochs without improvement of val loss to stop training


args=load_args()
logger=get_logger(args)
logger.info(args)
seed = args.seed
if args.local:
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    set_random_seed(seed)
    # if args.model==1 and args.gpu>=0:
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)

# batch_size = 128   #卧槽，这么大。                # Number of sequences in a batch
# flownum=30
# flowlen=2
# context_size = 64     #这个确实没有必要这么大。             # Size of the RNN hidden vector
# mark_embedding_size = 32      #才两个事件，确实没有必要这么大。     # Size of the mark embedding (used as RNN input)
rnn_type = "GRU"   #这个倒是无所谓。                # What RNN to use as an encoder {"RNN", "GRU", "LSTM"}
time_scale = args.time_scale

device=torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu>=0 else "cpu")
args.device=device
params = {
            'dataset_name': args.data,
            'batch_size': args.bs,
            'context_size': args.hdim,
            'mark_embedding_size': args.hdim,
            'rnn_type': rnn_type,
            'time_scale': time_scale,
            "device":device,
            # 'num_mix_components': num_mix_components,
            'pro_path':args.pro_path,
    'data_path': args.data_path,
    "seqlen":args.seqlen,
            "args":args

        }

train_dataset(params,logger,args)#