import torch.nn as nn
import torch
import dpp
from dpp.models.branch import *
from torch.nn.functional import one_hot
from torch.distributions import Categorical
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        d_k=Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)#
        context = torch.matmul(attn, V)#
        return context, attn#

class MultiHeadAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadAttention, self).__init__()
        self.args=args
        d_model,d_k,d_v,n_heads=args.hdim,args.hdim//args.headsnum,args.hdim//args.headsnum,args.headsnum
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)#注意力层之后不但要进行残差连接还要进行laynorm。但是残差是在forward中做的，所以不在这里定义，而且也没有残差这个函数，因为其就是手动加而已。

    def forward(self, Q, K, V, attn_mask):
        args=self.args#Q,K,V[bsize,seqlen,hdim]attn_mask[bsize,seqlen,seqlen]那我们这里相当于是一个上对角矩阵，其余全是0
        d_model,d_k,d_v,n_heads=args.hdim,args.hdim//args.headsnum,args.hdim//args.headsnum,args.headsnum
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        #至此，已经完成了特征分割。
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        #又是复制，这个repeat比前面那个expand更加通用。
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)#
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)#[bsize,seqlen,headsnum,hdim]->[bsize,seqlen,hdim]对融合之后的注意力做一个线性变换。
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,args):
        super(PoswiseFeedForwardNet, self).__init__()
        #我印象中，那个d_ff就是4倍的关系，所以我这里直接也是4倍的关系。
        self.args = args
        d_model, d_k, d_v, n_heads = args.hdim, args.hdim, args.hdim, args.headsnum
        d_ff=d_model*4
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)#全连接层怎么变成了这个。其实就是d_ff个kernel，一个kernel处理一个序列的第一维度。
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)#发现了，上述等价于线性层，这个人有病。
        self.layer_norm = nn.LayerNorm(d_model)#残差和标准化层。

    def forward(self, inputs):#[1,5,512]
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))#有毒，有的人喜欢使用nn的relu，有的人喜欢使用F的。
        output = self.conv2(output).transpose(1, 2)#上面[batch_size,emb_size,seq_len]->[batch_size,seq_len,emb_size]*[emb_size,d_ff]=[batch_size,seq_len,d_ff]
        return self.layer_norm(output + residual)#[batch_size,seq_len,d_ff]*[d_ff,d_model]=[batch_size,seq_len,d_model]

class Transformer_Layer(nn.Module):#用于增强表达能力，普通transformer，建议直接抄，因为有norm之类的操作，这里不需要前面无法注意到后面mask这种限制。
    def __init__(self,args):
        super(Transformer_Layer, self).__init__()#注意力层和线性层。
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, enc_inputs, enc_self_attn_mask):#encoder中有这个attn_mask真的是一个迷。
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class MANet(nn.Module):
    def __init__(self,args,inputdim,hdim,headsnum=4):
        super(MANet,self).__init__()
        self.inputdim=inputdim
        self.hdim=hdim
        self.headsnum=headsnum
        self.headsdim=hdim//headsnum
        self.transformer=nn.ModuleList()
        for i in range(args.layers):
            self.transformer.append(Transformer_Layer(args))
        #好像就完成了。
        self.wq=nn.Linear(args.hdim,hdim)#中间一律是hdim，哎一个问题是，我这样初次实验，不知道效果会不会好啊，到时候又要改了。
        self.wk=nn.Linear(inputdim,hdim)#这个是mdim+1相当于有时间这个维度。
        self.wv=nn.Linear(inputdim,hdim)#这里没有多头机制。其实就是类似于hdim必须是headsnum的整数倍。
        self.args=args
    def forward(self,q,k,masks):
        #q[bsize,M,hdim]k[bsize,seqlen,hdim]
        #masks[bisze,seqlen]
        #需要重新做一个mask,这个mask
        #而是有目的性的q,k,v。
        seqlen=k.shape[1]
        masks_tran=masks.unsqueeze(1).repeat(1,seqlen,1)#[bsize,seqlen,seqlen]#为transformer而准备的。
        for i in range(len(self.transformer)):
            k,_=self.transformer[i](k,masks_tran.type(torch.bool))#这里的那个attn我们不要。

        qx=self.wq(q)#[bsize,inputdim]->[bsize,M,hdim]
        kx=self.wk(k)#[bsize,seqlen,inputdim]->[bsize,seqlen,hdim]
        vx=self.wv(k)#一个问题，这个时候要不要包括自己呢？不需要吧，只是聚合历史信息而已。
        #然后开始attention即可。多头的话，就更加麻烦了，下面是多头步骤。
        headsnum=self.headsnum
        headsdim=self.headsdim
        hdim=self.hdim
        bsize=len(q)
        qx=qx.view(bsize,-1,headsnum,headsdim).permute(2,0,1,3)#[headsnum,bsize,M,headsdim]
        kx=kx.view(bsize,-1,headsnum,headsdim).permute(2,0,3,1)#[headsnum,bsize,headsdim,seqlen]
        vx=vx.view(bsize,-1,headsnum,headsdim).permute(2,0,1,3)#[headsnum,bsize,seqlen,headsdim]
        if self.args.norm==0:#说明不需要norm
            qk=torch.matmul(qx,kx)#[headsnum,bsize,M,seqlen]#然后可以对最后一维进行softmax即可。
        #这个玩意不知道有没有影响，要不要除以呢。试过了
        else:
            qk=torch.matmul(qx,kx)/ np.sqrt(headsdim)#[headsnum,bsize,M,seqlen]#然后可以对最后一维进行softmax即可。
        num_marks=qk.shape[2]
        masks=masks.unsqueeze(0).repeat(headsnum,1,1).unsqueeze(2).repeat(1,1,num_marks,1)#[headsnum,bsize,M,seqlen]
        #然后再进行填充即可。
        qk.masked_fill_(masks.type(torch.bool), -1e9)#大概就是这样搞定了，这下应该就算搞定了，还有一个。
        qk=torch.softmax(qk,dim=-1)#[headsnum,bsize,1,seqlen]
        qkv=torch.matmul(qk,vx)#[headsnum,bsize,1,headsdim]
        qkv=qkv.permute(1,2,0,3).reshape(bsize,-1,hdim)#[bsize,1,hdim]对了忘了说一个事情，这里的1不一定是一，因为我们有多个mark，可能会搞很多次的。


        return qkv#得到了[bsize,seqlen,hdim]个查询结果。这些查询结果会输入到线性层中去。
class TimeEmb(nn.Module):
    def __init__(self,args):
        super(TimeEmb,self).__init__()
        self.args=args
        self.w=nn.Parameter(torch.rand(1,1,args.hdim))#[1,hdim]
        self.b=nn.Parameter(torch.rand(1,1,args.hdim))#[1,hdim]

    def forward(self,t):
        #t[bsize,seqlen,1]
        return torch.cos(self.w*t+self.b)#好像就没事了。[bsize,seqlen,1]+[1,1,hdim]进行广播，没有毛病，就这样结束了。

class RecurrentTPP(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """
    def __init__(
        self,
        args,
        num_marks: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
            mean_cum=0.0,
            std_cum=1.0,
        mean_inter_time:float=0.0,
        std_inter_time:float=1.0,
        context_size: int = 32,
        mark_embedding_size: int = 32,
        rnn_type: str = "GRU",
        flownum:int =30,
        flowlen:int=2,
        # device=None,
    ):
        super().__init__()#这个init和原来的代码一模一样，作者没有修改。
        self.args=args
        self.num_marks = num_marks
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.mean_inter_time = mean_inter_time
        self.std_inter_time = std_inter_time
        self.mean_cum = mean_cum
        self.std_cum = std_cum
        self.context_size = context_size
        self.mark_embedding_size = mark_embedding_size
        self.flownum=flownum
        self.flowlen=flowlen
        # self.device=device
        self.num_features = self.mark_embedding_size
        self.mark_embedding = nn.Embedding(self.num_marks, self.mark_embedding_size)
        self.markp = MLP(self.context_size,self.context_size, self.num_marks)  #我们才不像下面那样，就简单使用一个linear呢，我们复杂一点。
            # self.mark_linear = nn.Linear(self.context_size, self.num_marks)#仅仅用一层就分类是什么事件，事件类型数量少，这样当然没有什么问题。

        self.rnn_type = rnn_type
        self.context_init = nn.Parameter(torch.zeros(self.num_features))  # initial state of the RNN
        # self.rnn = nn.Linear(self.num_features, self.context_size)
        self.rnn=MANet(args,self.num_features, self.context_size,headsnum=args.headsnum)#一个非常可恶的点是，我们初次弄这个，到时候结果还不一定对。似乎感觉应该先做另外一个东西，老师才会安心啊。
        # self.timepara=TimePara(self.context_size,self.context_size,flownum*flowlen,flownum,flowlen,self.num_marks)
        #
        self.headsnum=args.headsnum
        hdim=context_size
        self.linear_mark=nn.Parameter(torch.randn(num_marks,hdim))#各自都有一份。
        if args.timeencode==1:#vector
            self.timeemb=TimeEmb(args)

    def get_features(self, batch):#这个也和原来的代码一样。
        """
        Convert each event in a sequence into a feature vector.

        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).

        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        """
        #我们这次采用据当前最近的方式来搞。
        times=batch.inter_times.clone()#[bsize,seqlen]需要先逆转，然后cumsum。
        seqlen=times.shape[1]
        masks=batch.masks#[bsize]#需要将最后一个位置变为0啊。
        bsize=len(times)
        times=torch.cumsum(times,dim=1)-times[:,[0]]#[bsize,seqlen]为什么变成累加和的形式，因为希望可以任意注意到任意一个位置的东西，而不需要注意到旁边的来推测时间。
        #还有一个减号，为的就是搞一个偏置。一个问题，这里好像需要将seqlen删除1个？因为根本不需要注意到待预测的东西。
        times=times[:,:-1]#不要最后一个，因为没用，然后和下面的第一个填充一放，就够了。
        #所以其实只用前mask-1个。
        features = times.unsqueeze(-1)  # (batch_size, seq_len-1, 1)
        # features = torch.log(times + 1e-8).unsqueeze(-1)  # (batch_size, seq_len-1, 1)

        if self.args.norm==1:#说明不需要norm
            features = (features - self.mean_cum) / self.std_cum#使用了这个东西。但是这个东西说实话对现在的我没有什么用啊。这个东西到时候我计算过一遍，可以粗略算一下。
        #这个东西到底要不要归一化。
        # features = (features - self.mean_cum) / self.std_cum#使用了这个东西。但是这个东西说实话对现在的我没有什么用啊。这个东西到时候我计算过一遍，可以粗略算一下。

        if self.num_marks > 1:
            mark_emb = self.mark_embedding(batch.marks[:,:-1])  # (batch_size, seq_len-1, mark_embedding_size)
            # features = torch.cat([features, mark_emb], dim=-1)#这里不再使用concat了，直接将最后一个给替换掉。
            if self.args.timeencode==0:
                mark_emb[:,:,[-1]]=features#不知道能不能直接这样赋值，因为mark_emb不知道算不算叶子节点。
            else:
                mark_emb=mark_emb+self.timeemb(features)
            features =mark_emb
        #现在还是要对上述进行改造。
        context_init = self.context_init[None, None, :].expand(bsize, 1, -1)  # (batch_size, 1, hdim)
        #从而concat到上面去。
        features=torch.cat([context_init,features],dim=1)#[bsize,seqlen,hdim]
        #现在是要分qx,kx
        allm=torch.arange(self.num_marks,device=features.device).type(torch.int64).unsqueeze(0).repeat(bsize,1)#[1,num_marks]->[bsize,num_marks]
        qx=self.mark_embedding(allm)#[bsize,numm,hdim]
        #一个问题是没有添加
        #好像就是这两个了？不是，至于那个时间特征，大可不必吧。反正都是0，也没有什么意义啊。
        masks=batch.masks#[bsize]
        #这个masks需要进行填充啊，即填充到[bsize,seqlen]，这个真是麻烦的，有点难度啊。
        #难道需要循环。
        masks_fill=torch.ones_like(batch.inter_times)#[bsize,seqlen]
        for i in range(bsize):
            masks_fill[i,:masks[i]]=0#这里使用的是前mask个都要，这真的对吗？对的，因为有一个填充的，说句实话，第一个填充的，我是不太想要的。但是又不好搞，除非说bs=1，否则还是有些难办，就这样吧。
        return qx,features,masks_fill  # (batch_size, seq_len, num_features)

    def get_context(self, qx,features,masks_fill) :#使用gru，当然也是一样的。（目前搞第一个好像有点不行啊）
        #qx[bsize,marksnum,mdim]features[bsize,seqlen,hdim]masks_fill[bsize,seqlen]
        context = self.rnn( qx,features,masks_fill)#[bsize,marksnum,hdim]#得到了这些东西之后当然就是可以进行接下来的动作啦。
        return context

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raise NotImplementedError()
    def log_prob_trunc(self,#截断，即只输入一部分历史来预测下一个，他们以前的是每一个序列作为一个batch，我觉得可以不用这样，打乱来他不香吗？
                 batch: dpp.data.Batch) -> torch.Tensor:  # 原作者，这里得到了context之后一个linear得到了时间间隔分布的参数，用另外一个linear得到了是哪一个事件，相当于就是独立建模
        """Compute log-likelihood for a batch of sequences.#所以，作者在这里当然改了，改成了时间依赖于事件类型的结果。

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """
        flownum=self.flownum
        flowlen=self.flowlen
        num_marks=self.num_marks
        qx, features, masks_fill = self.get_features(batch)  # [bsize,seqlen,hdim]
        bsize=len(qx)#trunc的情况下，这个bsize是不一定的，因为一整个序列都会被拿来训练。[]
        brange=list(range(bsize))
        masks=batch.masks-1
        context = self.get_context( qx,features,masks_fill)  # [bsize,marksnum,hdim]#有了这些东西之后，就可以分别出击了。接一些线性层，查看以下效果。
        #事件类型分类时是否共享权值。
        share=True
        if not share:
            marks_linear=self.linear_mark.unsqueeze(0).repeat(bsize,1,1)#[bsize,marksnum,hdim]
            markslogit=context*marks_linear#[bsize,marksnum,hdim]
            markslogit=markslogit.sum(dim=-1)#搞定了。然后再softmax即可。
            mark_logits=torch.log_softmax(markslogit,dim=-1)#[bsize,marksnum]这里省略了1，没有搞。
        else:
            marks_linear=self.linear_mark[0].unsqueeze(0).unsqueeze(0)#[1,1,hdim]
            markslogit=context*marks_linear#[bsize,marksnum,hdim]
            markslogit=markslogit.sum(dim=-1)#搞定了。然后再softmax即可。
            mark_logits=torch.log_softmax(markslogit,dim=-1)#[bsize,marksnum]这里省略了1，没有搞。
        trunc=True#那么只剩下这个timepara了。
        # weight, bias = self.timepara(context)  # cdf的参数，两个都是[bsize,seqlen,nummarks*flowlen*flownum]
        # 先构建好这个网络，是临时生成的。
        # cdf = CDF(weight, bias,flownum,flowlen,num_marks,self.mean_inter_time,self.std_inter_time)  # 然后就可以调用其东西了。
        inter_times = batch.inter_times.clamp(1e-10)#我们其实不需要clamp下面解释了，但是这里仍然这么做，其实只是无所谓而已，相当于clone了。
        if trunc:#
            #上面的那个形状应该是[bsize,seqlen]，同理，我们也是只要最后一个。
            inter_times=inter_times[brange,masks].unsqueeze(1)#[bsize,1]
        inter_times_marks=inter_times.unsqueeze(2).repeat(1,1,num_marks)#这个是完全仿照原作者做法的。
        #[bsize,1,m]
        # flowindex=batch.flowindex.unsqueeze(-1)
        # flowindex=flowindex.unsqueeze(-1).repeat(1,1,num_marks).reshape(-1,1)
        # F,f = cdf.pdf(inter_times_marks, flowindex,training=self.training)  # [bsize,seqlen,num_marks]得到了是这个时间的概率。
        # t=cdf.sample()#测试暂时通过
        inter_time_dist = self.get_inter_time_dist(context)  # context要求是[bsize,seqlen,hdim],而我的是[bsize,m,hdim]这个其实并不关键好像，无非就是得到分布的参数，然后构建那个lognormal分布。
        log_p = inter_time_dist.log_prob(#每一个位置都有3个参数，那么参数就是[bsize,seqlen,3*m]，但是我这里本身就有m了，会是[bsize,m,3*m]需要修改线性层，变成[bsize,m,3]->[bsize,1,m*3]方法可以直接是将变成seqlen为1就行了。
            inter_times_marks)#[bsize,seqlen,m]分布为[bsize,1,m,3]输入的时间为[bsize,1,m]没有毛病。输出也为[bsize,1,m]各自的logp，下面会挑选。
        # log_p=torch.log(f+1e-8).view(bsize,-1,num_marks)#这个不知道有没有对啊，应该是没有问题的。
        marks=batch.marks
        if trunc:
            marks=marks[brange,masks]#[bsize,1]
        multi_marks = one_hot(marks.unsqueeze(1),num_classes=self.num_marks).float()  # (batch_size, seq_len, num_marks)总之就也是增加一维的意思。那个class我猜其实是可以不写的。
        pos_log_p = log_p * multi_marks  # (batch_size, seq_len, num_marks)#这个其实就等价于索引啊，这个log_p是得到了所有事件类型下当前时间的概率，即p(t|m=k,h)，然后这里乘以了一个one-hot，就相当于取出了那个正确的k,当然是希望这个东西越大越好。其实有一个点，如果只是希望这个越大越好的话，那么基于其他事件类型的分布，就没有利用上，这个作者相当于是白做了，说真的。
        mark_logits = mark_logits.unsqueeze(1)  # [bsize,1,num_marks]
        tot_nll = (pos_log_p + mark_logits * multi_marks).sum(-1) # (batch_size, seq_len)
        tot_nll = tot_nll.sum(-1)   # (batch_size)
        #

        mark_dist = Categorical(logits=mark_logits)  #上面不是计算了mark_logits嘛？还要分类干嘛呢？别管，就和他一样即可。
        log_mark = mark_dist.log_prob(marks.unsqueeze(1))  # 这三步都是懵逼的。[bsize,seqlen]取出来的应该就是概率，和mark_logits的差别也就在于上面那个*multi_marks了吧。实在看不懂。
        #
        # mark_class_joint = (log_p + mark_logits).argmax(
        #     -1).float()  # (batch_size, seq_len)#这个是联合概率。
        mark_class_joint =  mark_logits.argmax(
            -1).float()  #
        log_time = pos_log_p.sum(-1)  # [bsize,seqlen,2].sum()其实就是得到p(t|m=k,h)这个k是正确的k。然后mask就是取出需要的。
        # 那么问题来了，它到底在干什么，其实就是在得到log_time,log_mark，想要有分别的指标而已，到时候更好评价。
        return tot_nll, log_time, log_mark,  mark_class_joint#大概是改完了。还剩下前面的batch设计就完事了。

