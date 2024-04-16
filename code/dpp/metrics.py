from sklearn import metrics
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import dpp
sns.set_style('whitegrid')


def mark_metrics(lengths, mark_pred, mark_gt,logger):
    mark_gt_f=mark_gt.cpu().numpy()
    mark_pr_f=mark_pred.squeeze(1).cpu().numpy()
    acc = metrics.accuracy_score(mark_gt_f, mark_pr_f) * 100
    f1_micro = metrics.f1_score(mark_gt_f, mark_pr_f, average='micro') * 100
    f1_macro = metrics.f1_score(mark_gt_f, mark_pr_f, average='macro') * 100
    f1_weighted = metrics.f1_score(mark_gt_f, mark_pr_f, average='weighted') * 100    

    # print(f"f1_micro (Acc) : {f1_micro:.4f}")
    # print(f"f1_macro       : {f1_macro:.4f}")
    # print(f"f1_weighted    : {f1_weighted:.4f}")
    logger.info(f"f1_micro (Acc) : {f1_micro:.4f}")
    logger.info(f"f1_macro       : {f1_macro:.4f}")
    logger.info(f"f1_weighted    : {f1_weighted:.4f}")

def nll_metrics(time_nll, mark_nll, total_nll, total_count, interval,logger):
    
    time_nll = np.concatenate(time_nll)
    mark_nll = np.concatenate(mark_nll)
    total_nll = np.concatenate(total_nll)
    interval = np.concatenate(interval)
    
    total_nll_by_time = total_nll / interval
    
    tot_time_nll = time_nll.sum() / total_count
    tot_mark_nll = mark_nll.sum() / total_count
    tot_nll = total_nll.sum() / total_count
    tot_nll_by_time = total_nll_by_time.sum() / total_count
    
    # print(f"Time_NLL       : {tot_time_nll:.4f}")
    # print(f"Mark_NLL       : {tot_mark_nll:.4f}")
    # print(f"NLL            : {tot_nll:.4f}")
    # print(f"NLL/TIME       : {tot_nll_by_time:.4f}")
    logger.info(f"Time_NLL       : {tot_time_nll:.4f}")
    logger.info(f"Mark_NLL       : {tot_mark_nll:.4f}")
    logger.info(f"NLL            : {tot_nll:.4f}")
    logger.info(f"NLL/TIME       : {tot_nll_by_time:.4f}")

class Ba():
    def __init__(self,batch,device):
        self.inter_times = batch[0].to(device)
        self.marks = batch[1].to(device)
        # self.flowindex = batch[2].to(device)
        self.masks=batch[2].to(device)
def aggregate_loss_over_dataloader(model, dl, eval_mode=False,logger=None,params=None):#牛批，这个竟然是dpp内置的。不对，原来的作者并没有这个，而是新作者加上去的。
    device=params["device"]
    num_seqs=params["num_seqs"]
    if eval_mode:#这个是比较详细的评估，专门用于训练好了模型之后，才会动用这个选项，否则是进入else选项。
        total_loss = 0.0
        total_count = 0
        
        time_nll = []
        mark_nll = []
        surv_nll = []
        total_nll = []
        
        lengths = []
        mark_pred = []
        mark_gt = []
        
        interval = []
        
        with torch.no_grad():
            for batch in dl:
                batch=Ba(batch,device)
                tot_nll, log_p, log_mark, mark_pred_batch = model.log_prob_trunc(batch)
                total_loss += (-1)*tot_nll.sum().item()
                total_count += len(batch.marks)#我这样相当于是求每一个位置的平均损失，而之前不知道是不是每一个序列的平均损失。

                time_nll.append(-log_p.sum(-1).detach().cpu().numpy()) #(batch_size,)
                mark_nll.append(-log_mark.sum(-1).detach().cpu().numpy()) #(batch_size,)
                total_nll.append(-tot_nll.detach().cpu().numpy()) #(batch_size,)
                mark_pred.append(mark_pred_batch)  # [bsize,1]这个是预测的逻辑值。
                masks=batch.masks-1
                brange=list(range(len(mark_pred_batch)))
                mark_gt.append(batch.marks[brange, masks])  # [bsize,1]这个就是gound truth的意思。
        time_nll=np.concatenate(time_nll).sum()
        mark_nll=np.concatenate(mark_nll).sum()

        logger.info("loss:{}".format(total_loss/total_count))
        logger.info("time:{}".format(time_nll / total_count))
        logger.info("mark:{}".format(mark_nll / total_count))
        logger.info("loss:{}".format(5*total_loss/num_seqs))#这里乘以5是因为测试集占比20%
        logger.info("time:{}".format(5*time_nll / num_seqs))
        logger.info("mark:{}".format(5*mark_nll / num_seqs))#经过测试之后，其实发现两者差不多啊，搞鬼。


        mark_pred=torch.cat(mark_pred,dim=0)#[total_count,1]
        mark_gt=torch.cat(mark_gt,dim=0)#[total_count,1]
        mark_metrics(total_count,mark_pred,mark_gt,logger)

    else:#奇怪，
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():#不要梯度，这个是完全没有问题的，但是我们那个就有问题了。
            for batch in dl:
                batch=Ba(batch,device)
                tot_nll, log_p, log_mark, mark_pred = model.log_prob_trunc(batch)#还是一毛一样。
                total_loss += (-1)*tot_nll.sum().item()#但是我们发现，其只需要tot_nll，比较这个即可。[bsize]sum,-1那么就是越大越好了。
                total_count += len(batch.marks)#一共有多少个序列。
                
    return 5*total_loss/num_seqs


# Code from https://colab.research.google.com/github/shchur/shchur.github.io/blob/gh-pages/assets/notebooks/tpp2/neural_tpp.ipynb
def sampling_plot(model, t_end, num_seq, dataset,pro_path,name):
    with torch.no_grad():
        bs=64

        sampled_batch = model.sample(t_end=t_end, batch_size=num_seq)#画图，哈哈哈，有点东西，这里应该就是和我之前尝试过的一样，采样到t_end，然后采样若干个序列，看一下结果。
        real_batch = dpp.data.Batch.from_list([s for s in dataset])#怎么又来这一个？这个dataset是所有数据集，包括训练，测试，验证。他这里就是想要比较真实采样的数据集，和模型得到的数据集之间的差距，可以的。

        fig, axes = plt.subplots(figsize=[8, 4.5], dpi=200, nrows=2, ncols=2)#sub子绘画区，两行两列，dip=200,学到了。
        #fig应该是整张图，axes是某一个轴，这里有4个。
        for idx, t in enumerate(real_batch.inter_times.cumsum(-1).cpu().numpy()[:8]):#这里是0,0第0个轴开始绘制。真实数据的[0,T]分布。
            axes[0,0].scatter(t, np.ones_like(t) * idx, alpha=0.5, c='C2', marker="|")#这里相当于是画10000多条横线。
        axes[0,0].set_title("Arrival times: Real event sequences", fontsize=7)#原来还会控制字体大小。上面这个c2好像是绿色。
        axes[0,0].set_xlabel("Time", fontsize=7)#但是我们查看的时候，好像并没有1万条线段，只有7条怎么会是。好像明白了，下面的yticks的原因嘛？
        axes[0,0].set_ylabel("Different event sequences", fontsize=7)
        axes[0,0].set_yticks(np.arange(8))#[0,7]才会显示。
        axes[0,0].xaxis.offsetText.set_fontsize(7)#竟然还有这个，

        for idx, t in enumerate(sampled_batch.inter_times.cumsum(-1).cpu().numpy()[:8]):#又是这个。但是这个是我们采样得到的数据。
            axes[0,1].scatter(t, np.ones_like(t) * idx, alpha=0.5, c='C3', marker="|")#换了颜色，什么鬼，颜色变成红色了。
        axes[0,1].set_xlabel("Time", fontsize=7)
        axes[0,1].set_title("Arrival times: Sampled event sequences", fontsize=7)
        axes[0,1].set_ylabel("Different event sequences", fontsize=7)
        axes[0,1].set_yticks(np.arange(8))
        axes[0,1].xaxis.offsetText.set_fontsize(7)#这些都是一模一样的。总之是文本对象，这个offsetText应该指代的就是xtick上面的文字？然而好像并不是，这个东西目前好像并没有发现什么作用。

        sample_len = sampled_batch.mask.sum(-1).cpu().numpy()#这里又开始画这两个东西了，依然是左边为真实，右边为采样。
        real_len = real_batch.mask.sum(-1).cpu().numpy()#这个是mask相加，即tend之前有多少个事件。从而1个事件的有多少个序列，这样可以搞一个分布。
        
        axes[1,0].set_title("Distribution of sequence lengths", fontsize=7)
        q_min = min(real_len.min(), sample_len.min()).astype(int)#长度最小值，因为这里是要画在一个轴里了，上面一个轴一个。
        q_max = max(real_len.max(), sample_len.max()).astype(int)#长度最大值。
        axes[1,0].hist([real_len, sample_len], bins=30, alpha=0.9, color=["C2","C3"], range=(q_min, q_max), label=["Real data", "Sampled data"]);
        axes[1,0].set_xlabel(r"Sequence length", fontsize=7)#注意到，上面是hist，频率分布直方图。和我之前画的完全不一样啊。
        axes[1,0].set_ylabel("Frequency", fontsize=7)
        
        sampled_marks_flat = []#flat是平的意思。
        real_marks_flat = []
        
        for i, each in enumerate(sampled_batch.marks):
            sampled_marks_flat.append(sampled_batch.marks[i, :sampled_batch.mask[i].sum().int()].detach().cpu().numpy())
        #取出每一个序列的mark。然后组织成为列表。
        for i, each in enumerate(real_batch.marks):#同上，这个是真实数据。
            real_marks_flat.append(real_batch.marks[i, :real_batch.mask[i].sum().int()].detach().cpu().numpy())

        sampled_marks_flat = np.concatenate(sampled_marks_flat)#他这里不需要在乎序列了，直接统计事件类型的种类分布即可。
        real_marks_flat = np.concatenate(real_marks_flat)

        axes[1,1].set_title("Distribution of marks", fontsize=7)
        unique, counts = np.unique(np.asarray(sampled_marks_flat), return_counts=True)#不同种类，各自有多少个，和count估计也差不多。
        unique_0, counts_0 = np.unique(np.asarray(real_marks_flat), return_counts=True)#同上。
        
        q_min = min(unique.min(), unique_0.min()).astype(int)#又是这个，有可能某一方压根就没有发生某一个事件类型，此时，这个qminmax就会起作用了。
        q_max = max(unique.max(), unique_0.max()).astype(int)#同上。
        axes[1,1].hist([real_marks_flat, sampled_marks_flat], alpha=0.9, color=["C2","C3"], range=(q_min, q_max), label=["Real data", "Sampled data"]);
        axes[1,1].set_xlabel(r"Marks", fontsize=7)#又是这个，但是这里怎么没有写那个bins了，为啥啊。
        axes[1,1].set_ylabel("Frequency", fontsize=7)

        axes[1,0].legend(ncol=1, fontsize=7)
        axes[1,1].legend(ncol=1, fontsize=7)

        axes[1,1].yaxis.offsetText.set_fontsize(7)#我去这是为啥啊，又来了这个。

        for ax in np.ravel(axes):#我去，这个是什么鬼。这个其实就等价于flatten。
            ax.tick_params(axis='x', labelsize=7)#卧槽，竟然会有这么多骚操作。
            ax.tick_params(axis='y', labelsize=7)

        fig.tight_layout()#紧布局。看来真的得好好学习他这里的几招。
        figpath=pro_path+"/figures/{}.png".format(name)
        plt.savefig(figpath,dpi=300)
        plt.show()#作者没有加这句，在Jupyter上面是可以，在pycharm里面可不行啊。
