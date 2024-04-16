import dpp
import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from dpp.metrics import aggregate_loss_over_dataloader, sampling_plot#这两个东西，我有点忘记了，干什么用的了。

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
def load_data(params,args):
    dataset_name = args.data
    batch_size = args.bs
    device=args.device
    pro_path=args.pro_path
    data_path=args.data_path
    seqlen=args.seqlen

    train_seqs,val_seqs,test_seqs,num_seqs,num_marks,t_end = load_dataset(dataset_name,data_path)#好家伙，原来是使用了dpp的读取数据，我记得我是看过这些api的。
    params["t_end"],params["num_seqs"]=t_end,num_seqs
    d_train = get_dataset(train_seqs,seqlen,num_marks=num_marks)#又是熟悉的感觉，不过，没有必要进去看了。
    d_val = get_dataset(val_seqs,seqlen,num_marks=num_marks)
    d_test = get_dataset(test_seqs,seqlen,num_marks=num_marks)

    dl_train = d_train.get_dataloader(batch_size=batch_size, shuffle=True)#我们这里不进行shuffle。也希望一个一个进行训练。
    dl_val = d_val.get_dataloader(batch_size=batch_size, shuffle=False)
    dl_test = d_test.get_dataloader(batch_size=batch_size, shuffle=False)#都没有什么问题，然后那个划分比例，就是默认的6/2/2，和它生成的那个其实是一模一样的。
    #比较关心的无非是时间，时间已经变成了间隔，而且最后一个生存时间还加上了，所以其实序列长度变长了1个。
    return  d_train, d_val, d_test, dl_train, dl_val, dl_test

def build_model(d_train, params):
    context_size = params['context_size']#数据集以及参数，数据集类型sequencedataset类型，说白了就是序列列表，每一个元素都是序列。
    mark_embedding_size = params['mark_embedding_size']
    rnn_type = params['rnn_type']
    device=params["device"]
    # num_mix_components = params['num_mix_components']


    mean_log_inter_time, std_log_inter_time = d_train.get_inter_time_statistics()#这个是获得时间间隔的统计数据，但是是log之后的，感觉我还是得看一看了，否则无法清楚对自己的数据集做了什么手脚。
    mean_cum, std_cum = d_train.get_cum_time_statistics()#这个是获得时间间隔的统计数据，但是是log之后的，感觉我还是得看一看了，否则无法清楚对自己的数据集做了什么手脚。
    # mean_cum, std_cum =-0.2266,6.6140 #这个是获得时间间隔的统计数据，但是是log之后的，感觉我还是得看一看了，否则无法清楚对自己的数据集做了什么手脚。

    #这个是间隔时间（除去了生存时间），然后log，然后mean，然后std。
    max_inter_time,mean_inter_time,std_inter_time=d_train.get_inter_time_max()

    model = dpp.models.LogNormMix(
        args=params["args"],
        num_marks=d_train.num_marks,
        mean_log_inter_time=mean_log_inter_time,
        std_log_inter_time=std_log_inter_time,
        mean_cum=mean_cum,
        std_cum=std_cum,
        mean_inter_time=mean_inter_time,
        std_inter_time=std_inter_time,
        context_size=context_size,
        mark_embedding_size=mark_embedding_size,
        rnn_type=rnn_type,
    )


    return model

class Ba():
    def __init__(self,batch,device):
        self.inter_times = batch[0].to(device)
        self.marks = batch[1].to(device)
        self.masks=batch[2].to(device)

def train_helper(model, dl_train, dl_val,logger,params,args):
    # Training config
    device=args.device
    bsize=args.bs
    num_seqs=params["num_seqs"]#
    regularization = args.regularization  # L2 regularization parameter
    learning_rate = args.lr  # Learning rate for Adam optimizer
    max_epochs = args.max_epochs  # For how many epochs to train
    display_step = args.display_step  # 原5，我们改成1       # Display training statistics after every display_step
    patience = args.patience  # 原50，我们改成3          # After how many consecutive epochs without improvement of val loss to stop training

    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)#参数减少。

    impatient = 0
    best_loss = np.inf#越小越好，所以这里设置为无穷大。
    best_model = deepcopy(model.state_dict())#他这里不是像我那样，每当有一个更好的模型就save，而是保存一个内存变量，最后再存。
    training_val_losses = []
    for epoch in range(max_epochs):#有一说一，恕我直言，他这个好像1个epoch就差不多了，后面一直训练，好像也没有什么太大的loss改进。
        model.train()
        losses = []
        for batch in dl_train:
            opt.zero_grad()#
            batch=Ba(batch,device)
            tot_nll, _, _, _ = model.log_prob_trunc(batch)#tot_nll，就是他们通常所说的那个nll，形状是[bsize]，下面求loss，mean了一下。
            loss = -tot_nll.sum()/bsize#为了对标，这里不应该使用均值，这样会出现一个问题，放大梯度，这样的话，学习率应该相应调整，如果不想调整，这里应该改为sum。
            loss.backward()#上面这个除以bsize，对于bsize比较大的时候好像会有点影响，但是数据量大的话，应该是微不足道的。
            losses.append(-tot_nll.sum().item())
            opt.step()

        model.eval()
        with torch.no_grad():
            loss_val = aggregate_loss_over_dataloader(model, dl_val,params=params)#竟然有了这个东西。我猜和上面那个log_prob差不多，类似于我们之前经常弄的pred
            training_val_losses.append(loss_val)#这个是每一个序列的平均值，注意，这个东西是越小越好。
            #他这里只是训练时候的评估，所以只需要得到totalloss，下面正式评估的时候，就会有各种类型的报告。
        if (best_loss - loss_val) < 1e-4:
            impatient += 1
            if loss_val < best_loss:
                best_loss = loss_val
                best_model = deepcopy(model.state_dict())
        else:
            best_loss = loss_val
            best_model = deepcopy(model.state_dict())
            impatient = 0

        if impatient >= patience:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break

        if epoch % display_step == 0:
            # print(f"Epoch {epoch:4d}: loss_train_last_batch = {loss.item():.1f}, loss_val = {loss_val:.1f}, , p_e = {impatient}")
            logger.info(f"Epoch {epoch:4d}: loss_training = {np.sum(losses)/(0.6*num_seqs):.1f}, loss_val = {loss_val:.1f}, , p_e = {impatient}")

    return best_model

def evaluation(model, dl_train, dl_val, dl_test,logger,params):
    device=params["device"]

    model.eval()

    # All training & testing sequences stacked into a single batch
    with torch.no_grad():
        # print('TRAIN')
        # _= aggregate_loss_over_dataloader(model, dl_train, eval_mode=True,logger=logger)
        # print('-'*30)
        # print('VAL')
        # _ = aggregate_loss_over_dataloader(model, dl_val, eval_mode=True,logger=logger)
        print('-'*30)
        print('TEST')
        _ = aggregate_loss_over_dataloader(model, dl_test, eval_mode=True,logger=logger,params=params)



def train_dataset(params,logger,args):
    print(params['dataset_name'])
    print('-'*50)
    print('Loading data..')
    d_train, d_val, d_test, dl_train, dl_val, dl_test = load_data(params,args)#导入数据，终于来了。
    print('-'*50)
    print('Building model..')
    model = build_model(d_train, params)#建立模型。
    model=model.to(args.device)
    print('-'*50)
    print('Training..')
    best_model = train_helper(model, dl_train, dl_val,logger,params,args)#这个其实就是训练。我怎么发现这个作者好像什么也没有做，作者lnm原来就是这么写的，不过好像没有condition上？
    model.load_state_dict(best_model)
    print('-'*50)
    print('Evaluation..')
    evaluation(model, dl_train, dl_val, dl_test,logger,params)#在所有数据集上进行评估。我发现，这个评估，在三个数据集上的结果都差不多。
    print('-'*50)
    # print('Sampling..')#我们没法sampling，这个实在没有办法。
    # t_end,num_seq=params["t_end"],params["num_seq"]
    # pro_path=params["pro_path"]
    # sampling_plot(model, t_end, num_seq, dataset,pro_path,params['dataset_name'])#这个可以看看其实，到时候我也画，不过不太好，我这个采样真的很不方便。
    print('-'*50)
    print('Saving model ..')
    pro_path=params["pro_path"]
    model_save_path = pro_path+'/models/{}-{}.pth'.format(params['dataset_name'].split('/')[-1],args.model)
    print(model_save_path)#存储模型，没哟与必要啊。
    # torch.save(model, model_save_path)
    torch.save(model.state_dict(), model_save_path)#


def sampling(params,logger,args):
    print(params['dataset_name'])
    print('-'*50)
    print('Loading data..')
    dataset, d_train, d_val, d_test, dl_train, dl_val, dl_test = load_data(params)#导入数据，终于来了。
    print('-'*50)
    print('Building model..')
    model = build_model(d_train, params)#建立模型。
    pro_path=params["pro_path"]
    model_save_path = pro_path+'/models/{}-{}.pth'.format(params['dataset_name'].split('/')[-1],args.model)
    # model=torch.load(model_save_path)
    if args.local:
        model.load_state_dict(torch.load(model_save_path,map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_save_path))
    print('-' * 50)
    print('Sampling..')  # 我们没法sampling，这个实在没有办法。
    t_end, num_seq = params["t_end"], params["num_seq"]
    pro_path = params["pro_path"]
    sampling_plot(model, t_end, num_seq, dataset, pro_path,
                  params['dataset_name'])  # 这个可以看看其实，到时候我也画，不过不太好，我这个采样真的很不方便。
    # print('-' * 50)
    # print('Saving model ..')
    # print(model_save_path)  # 存储模型，没哟与必要啊。
    # torch.save(model, model_save_path)
def eval_dataset(params,logger,args):
    print(params['dataset_name'])
    print('-'*50)
    print('Loading data..')
    dataset, d_train, d_val, d_test, dl_train, dl_val, dl_test = load_data(params)#导入数据，终于来了。
    print('-'*50)
    print('Building model..')
    model = build_model(d_train, params)#建立模型。
    pro_path=params["pro_path"]
    model_save_path = pro_path+'/models/{}-{}.pth'.format(params['dataset_name'].split('/')[-1],args.model)
    # model=torch.load(model_save_path)
    if args.local:
        model.load_state_dict(torch.load(model_save_path,map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_save_path))
    print('-' * 50)
    print('evaluating..')
    model.eval()#进入评估模式。
    #这里我们做两个评估。同时也要完成两件事情。此时会填满F。还是会有一个问题啊，很多mask的问题。
    # _ = aggregate_loss_over_dataloader(model, dl_val, eval_mode=True, logger=logger, device=params["device"])

    _ = aggregate_loss_over_dataloader(model, dl_test, eval_mode=True, logger=logger, device=args.device)
    print("done")



from pathlib import Path
from typing import List, Tuple, Union


import numpy as np
import torch
import torch.utils.data as tud
dataset_dir = Path(__file__).parents[3] / "data"
def pad(lis,seqlen):
    padl=seqlen-len(lis)
    li=lis+[0 for i in range(padl)]#为什么我的代码在2个数据集的时候会这么大错误呢？
    mask=len(lis)
    return li,mask#好像就搞定了。
def trunc(time_data,seq_len,time=False):
    #遍历每一个lis，开始划分即可。
    seqdata=[]
    masks=[]
    eventlen=[]
    for i in range(len(time_data)):#上面就是得到了若干个序列，每一个序列都有若干个事件。
        if time:
            line = list(np.diff([0]+time_data[i]))
        else:
            line = time_data[i]
        end = seq_len#卧槽，这里竟然还有一个seqlen，那看来很早就有人使用这个方法了啊，到时候我得提一嘴。
        eventlen.append(len(line))
        mins=min(seq_len-1,len(line))
        for i in range(mins):#这里需要填充，我们使用的是向前填充，从而有一个好处，无需mask就行，只看最后一个。
            end=i+1#
            sseq=line[:end]
            sseq,mask=pad(sseq,seq_len)
            seqdata.append(sseq)
            masks.append(mask)
        end=seq_len
        while end <= len(line):#突然才发现，这一段有点小问题，即没有预测第1个事件。
            start = end-seq_len
            seqdata.append(line[start:end])#原本元素是时间，现在是一串时间。而且密度很大，这个和我的做法是一毛一样的。
            end += 1
            mask=seq_len
            masks.append(mask)
    return seqdata,masks,eventlen

def load_dataset(name,data_path,train_size=0.6,val_size=0.2):
    if not name.endswith(".pkl"):
        name += ".pkl"
    path_to_file = data_path+"/data/"+ name
    dataset = torch.load(str(path_to_file))

    seqs=dataset["sequences"]
    num_seqs=len(seqs)
    num_marks=dataset["num_marks"]
    t_end=seqs[0].get("t_end")
    all_idx = np.arange(num_seqs)
    np.random.shuffle(all_idx)

    train_end = int(train_size * num_seqs)  # idx of the last train sequence
    val_end = int((train_size + val_size) * num_seqs)  # idx of the last val seq
    train_idx = all_idx[:train_end]
    val_idx = all_idx[train_end:val_end]
    test_idx = all_idx[val_end:]
    train_seqs=[]
    val_seqs=[]
    test_seqs=[]
    for i in range(num_seqs):
        if i in train_idx:
            train_seqs.append(seqs[i])
        elif i in val_idx:
            val_seqs.append(seqs[i])
        else:
            test_seqs.append(seqs[i])
    #就这样划分完毕了。
    return train_seqs,val_seqs,test_seqs,num_seqs,num_marks,t_end

def logdata(data):
    for i in range(len(data)):
        #对data中的每一个进行log化，然后还原。
        dd=data[i]
        dd["arrival_times"]=list(np.cumsum(np.log(1+np.diff(np.concatenate([[0], dd["arrival_times"]])))))
    return data

def get_dataset(sequences,seqlen,num_marks=2):
    sequences=logdata(sequences)#先这么解决了。
    #开始trunc了。
    seqs,_,_=trunc([seq.get("arrival_times") for seq in sequences],seqlen,time=True)#不需要开始时间和结束时间了，开始默认就是0.
    #如果不出意外的话，返回的应该是[bsize,seqlen]
    seqdata = np.array(seqs)  # 64个序列，然后一个序列大概有1000多个事件。这个其实不一定能够组织成array,需要建立在64个序列长度一样。事实证明，确实是一样，都是1397.
    marks,masks,eventlen=trunc([seq.get("marks") for seq in sequences],seqlen)#我们好像只需要这两个。
    marks=np.array(marks)#
    masks=np.array(masks)
    #其中各个东西的形状如下。seqdata[bsize,seqlen]marks也是，masks[bsize],eventlen[num_eventsequence]，或者就是叫做num_seqs。
    eventlen=np.array(eventlen)#eventlen是用来将同一个事件序列中的组成批来训练，这样有一个好处就是和原本的rnn那么就是完全对标了。
    eventlen=np.cumsum(eventlen)
    return SeqData(torch.from_numpy(seqdata).type(torch.float32),
                   torch.from_numpy(marks).type(torch.int64),
                torch.from_numpy(masks).type(torch.int64),torch.from_numpy(eventlen).type(torch.int64),num_marks=num_marks)

def my_collate(batch):
    #这个东西应该就是
    seqs=[]
    marks=[]
    masks=[]
    for seq,mark,mask in batch:
        seqs.append(seq)
        marks.append(mark)
        masks.append(mask)
    #然后将其concat即可。
    seqs=torch.cat(seqs)#默认好像就是第一个维度。
    marks=torch.cat(marks)
    masks=torch.cat(masks)#这个也是一维的。
    return seqs,marks,masks

class SeqData(tud.Dataset):
    def __init__(self,seqs,marks,masks,eventlen,num_marks=None):#其中eventlen
        self.seqs=seqs#这个是描述事件时间。
        self.marks=marks#这个是描述事件类型。
        self.masks=masks#描述填充，而且是填充在了后面。而且其仅仅是一个[bsize]的，即描述有多少个真的，而不是111000的形式。
        #我们接下来是要对这个东西进行拆分成更加标准的一个东西。
        self.num_marks=num_marks
        self.eventlen=eventlen
        #好像也就完了。
    def __len__(self):
        return len(self.eventlen)#改变了这个，其实也就意味着一切都变了。
    def __getitem__(self,ind):#我们这里设置了，marks是一定会有的。
        #我们这里搞那个index，相当于这里再搞一个flow
        if ind==0:
            start=0
        else:
            start=self.eventlen[ind-1]
        end = self.eventlen[ind]
        seq=self.seqs[start:end]#我明白了，不复存在就不复存在，无非是使用两份代码而已。但是真的可以快很多啊。[1,seqlen]
        mask=self.masks[start:end]
        return seq,self.marks[start:end],mask#没毛病了吧
    def get_dataloader(
            self, batch_size: int = 32, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        # if model_index==0:
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,collate_fn=my_collate
        )

    def get_inter_time_statistics(self):
        """Get the mean and std of log(inter_time)."""

        # all_inter_times = self.seqs#这里有点问题，包括进去了0之类的，而且对于一些前面的还重复了。
        allp=len(self.seqs)#seqs的shape应该是[allp,seqlen]然后靠eventlen来表示。
        listallp=list(range(allp))
        all_inter_times=self.seqs[listallp,self.masks-1]#mask[allp]好像刚好可以取出全部，这样得到的就是[allp]这个才是真正的intertimes。
        all_inter_times=torch.clamp(all_inter_times,min=1e-7)
        mean_log_inter_time = all_inter_times.mean()#log被我删除了，我们这里不再需要了。
        std_log_inter_time = all_inter_times.std()
        return mean_log_inter_time, std_log_inter_time
    def get_cum_time_statistics(self):
        """Get the mean and std of log(inter_time)."""
        #我们这次采用据当前最近的方式来搞。
        times=self.seqs.clone()#[bsize,seqlen]需要先逆转，然后cumsum。
        masks=self.masks#[bsize]#需要将最后一个位置变为0啊。
        bsize=len(times)
        times[list(range(bsize)),masks-1]=0#现在这样好像会改变原来的东西为了保险起见，直接clone。
        times=times[:,1:]#不要第一个间隔时间，没有用。
        times=torch.flip(times,dims=[1])#[bsize,seqlen]
        times=torch.cumsum(times,dim=1)#[bsize,seqlen]只要倒数mask-1个。而且是反过来的。
        times=torch.cat([torch.zeros(bsize,1,device=times.device),times],dim=-1)#
        times=torch.flip(times,dims=[1])#[bszie,seqlen]没有了毛病，那就是和原来一模一样了。前mask个就是需要的。mask-1个是k,mask是q。
        #但是为了并行着想，不能取，而是到时候使用mask来搞定。
        # all_inter_times = self.seqs#这里有点问题，包括进去了0之类的，而且对于一些前面的还重复了。
        all_inter_times=[]
        for i in range(len(times)):
            all_inter_times.append(times[i,:masks[i]-1])
        all_inter_times=torch.cat(all_inter_times)
        all_inter_times=torch.clamp(all_inter_times,min=1e-7)
        mean_log_inter_time = all_inter_times.mean()
        std_log_inter_time = all_inter_times.std()
        # mean_log_inter_time = all_inter_times.log().mean()
        # std_log_inter_time = all_inter_times.log().std()
        return mean_log_inter_time, std_log_inter_time
    def get_inter_time_max(self):#我们第一次，就直接平均分好了，所以我们只需要一个max即可。
        """Get the mean and std of log(inter_time)."""
        allp = len(self.seqs)  # seqs的shape应该是[allp,seqlen]然后靠eventlen来表示。
        listallp = list(range(allp))
        all_inter_times = self.seqs[listallp, self.masks-1]
        max_inter_time ,mean_inter_time,std_inter_time= all_inter_times.max(),all_inter_times.mean(),all_inter_times.std()
        return max_inter_time,mean_inter_time,std_inter_time
