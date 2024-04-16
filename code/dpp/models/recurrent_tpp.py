import dpp
import torch
import torch.nn as nn

from torch.distributions import Categorical

from dpp.data.batch import Batch
from dpp.utils import diff
from torch.nn.functional import one_hot
from util import Fs,seqlens,mse,test

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
        num_marks: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 32,
        mark_embedding_size: int = 32,
        rnn_type: str = "GRU",
    ):
        super().__init__()#这个init和原来的代码一模一样，作者没有修改。
        self.num_marks = num_marks
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size
        self.mark_embedding_size = mark_embedding_size
        if self.num_marks > 1:#事件类型多于1，在本作者代码中，全部都是大于1的。
            self.num_features = 1 + self.mark_embedding_size
            self.mark_embedding = nn.Embedding(self.num_marks, self.mark_embedding_size)
            self.mark_linear = nn.Linear(self.context_size, self.num_marks)#仅仅用一层就分类是什么事件，事件类型数量少，这样当然没有什么问题。
        else:
            self.num_features = 1
        self.rnn_type = rnn_type
        self.context_init = nn.Parameter(torch.zeros(context_size))  # initial state of the RNN
        self.rnn = getattr(nn, rnn_type)(input_size=self.num_features, hidden_size=self.context_size, batch_first=True)

    def get_features(self, batch: dpp.data.Batch) -> torch.Tensor:#这个也和原来的代码一样。
        """
        Convert each event in a sequence into a feature vector.

        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).

        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        """
        features = torch.log(batch.inter_times + 1e-8).unsqueeze(-1)  # (batch_size, seq_len, 1)
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time#使用了这个东西。
        if self.num_marks > 1:
            mark_emb = self.mark_embedding(batch.marks)  # (batch_size, seq_len, mark_embedding_size)
            features = torch.cat([features, mark_emb], dim=-1)
        return features  # (batch_size, seq_len, num_features)

    def get_context(self, features: torch.Tensor, remove_last: bool = True) -> torch.Tensor:#使用gru，当然也是一样的。
        """
        Get the context (history) embedding from the sequence of events.

        Args:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)
            remove_last: Whether to remove the context embedding for the last event.

        Returns:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size) if remove_last == False
                shape (batch_size, seq_len + 1, context_size) if remove_last == True

        """
        context = self.rnn(features)[0]
        batch_size, seq_len, context_size = context.shape
        context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        if remove_last:
            context = context[:, :-1, :]#老实说，我觉得这个还挺奇怪的。
        context = torch.cat([context_init, context], dim=1)
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

    
    def log_prob(self, batch: dpp.data.Batch) -> torch.Tensor:#原作者，这里得到了context之后一个linear得到了时间间隔分布的参数，用另外一个linear得到了是哪一个事件，相当于就是独立建模
        """Compute log-likelihood for a batch of sequences.#所以，作者在这里当然改了，改成了时间依赖于事件类型的结果。

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """
        
        features = self.get_features(batch)#[bsize,seqlen+1,hdim]
        context = self.get_context(features)#[bsize,seqlen+1-1+1,hdim]为什么少了一个，因为将最后一个删除了，为什么多了一个，因为有了context_init相当于要预测第一个事件。
        trunc=True
        if trunc:
            #我们只需要上面的最后一个即可。其实有一个rnn被丢弃了，上面有,remove。
            context=context[:,[-1]]#[bsize,1,hdim]
        inter_time_dist = self.get_inter_time_dist(context)#这个其实并不关键好像，无非就是得到分布的参数，然后构建那个lognormal分布。
        # inter_time_dist=inter_time_dist.to(features.device)
        inter_times = batch.inter_times.clamp(1e-10)#这个默认就是最小值，然后这个间隔时间别忘了，还没有log的。本文的get_features那里log，然后减去均值。
        if trunc:#
            #上面的那个形状应该是[bsize,seqlen]，同理，我们也是只要最后一个。
            inter_times=inter_times[:,[-1]]
        multi_marks = one_hot(batch.marks, num_classes=self.num_marks).float() # (batch_size, seq_len, num_marks)总之就也是增加一维的意思。那个class我猜其实是可以不写的。

        inter_times.unsqueeze_(-1) # (batch_size, seq_len, 1)
        inter_times = inter_times.expand(inter_times.shape[0],inter_times.shape[1],self.num_marks).clone() # (batch_size, seq_len, num_marks)
        #上面这个严格来说，inter以及marks其实都是【bsize,seqlen+1]，末尾多了一个T，至于那个mark就是填充了0而已。
        # print(inter_times.device)
        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len, num_marks)然后，上面的expand和repeat差不多，repeat是倍数，而expand是需要多少维，从不同角度描述罢了。
        # log_p = log_p.to(features.device)#本来不会先cpu然后又todevice的，但是那个分布好像不支持gpu啊。  # (batch_size, seq_len, num_marks)然后，上面的expand和repeat差不多，repeat是倍数，而expand是需要多少维，从不同角度描述罢了。

        #其实还是那个点，这里是要得到p(t|m=1,h)的值，所以上面要复制expand，num_marks个。上面这个直接就动手计算了。log_prob是内置的，本身就是 (batch_size, seq_len, num_marks)个分布，所以带入这么多个inter_time也很正常。
        pos_log_p = log_p * multi_marks # (batch_size, seq_len, num_marks)#这个其实就等价于索引啊，这个log_p是得到了所有事件类型下当前时间的概率，即p(t|m=k,h)，然后这里乘以了一个one-hot，就相当于取出了那个正确的k,当然是希望这个东西越大越好。其实有一个点，如果只是希望这个越大越好的话，那么基于其他事件类型的分布，就没有利用上，这个作者相当于是白做了，说真的。
        neg_log_p = log_p * (1 - multi_marks)#这个作者没有使用，不用管，你要说，这个东西代表了什么，代表了基于其他事件类型的logp。
        #
        # Survival probability of the last interval (from t_N to t_end).
        # You can comment this section of the code out if you don't want to implement the log_survival_function
        # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        # but the difference shouldn't be significant if you are working with long sequences.
        #上面那个生存概率也是重写的，在那个distrbutions下面有，也是按照标准流公式求解，F(y)，但是我没有搞懂，生存其实只有最后一个时间点需要，这里带入全部干嘛？可以看上面，取出来了最后一个，但是只是吐槽，浪费时间。
        #这里还是要反复梯形，intertime是包括最后一个T的，但是mask不包括。它上面，计算生存函数尽然这么搞，有没有搞错啊我去。他这里的生存概率很扯淡，是p(t>tend|m=mend,h)本来由于两个事件类型，会有两个生存，它上面就是直接选用了最后一个mark（不是填充的），我觉得很有问题啊。因为那个h其实就包括了mend的信息，它相当于只是索引了，选了其中一个作为生存概率。
        mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # (batch_size, seq_len, num_marks)现在才来这个，不过其实确实和我那个不一样，我那个是知道m=k的情况下，他这里是建模分布，我这里是建模某一个instance的具体概率值。
        tot_nll = (pos_log_p + mark_logits * multi_marks).sum(-1) #(batch_size, seq_len)
        tot_nll = tot_nll.sum(-1) #(batch_size)
        #上面第二哥这个我像就不用管了，后面有那个mask。然后那个加法其实就是联合分布了。然后上面这个就是加上最后一部分。
        mark_dist = Categorical(logits=mark_logits)#我去，竟然还有这个，这个是用来干嘛的。暂时没有看懂，上面不是计算了mark_logits嘛？还要分类干嘛呢？
        log_mark = mark_dist.log_prob(batch.marks)#又是一脸懵逼，[bsize,seqlen]取出来的应该就是概率，和mark_logits的差别也就在于上面那个*multi_marks了吧。实在看不懂。

        mark_class_joint = ((log_p + mark_logits).argmax(-1) * batch.mask).float() #(batch_size, seq_len)#这个其实是预测了下一个事件是什么，但是有一个问题，其基于了下一次发生事件的时间。从而有了logp，然后argmax，所以，它这个其实才是真正的选择了哪个事件，只是可惜，我们的损失函数并没有分类这一项。
        log_time = pos_log_p.sum(-1)#[bsize,seqlen,2].sum()其实就是得到p(t|m=k,h)这个k是正确的k。然后mask就是取出需要的。
        #那么问题来了，它到底在干什么，其实就是在得到log_time,log_mark，想要有分别的指标而已，到时候更好评价。
        return tot_nll, log_time, log_mark, mark_class_joint

    def log_prob_trunc(self,
                 batch: dpp.data.Batch) -> torch.Tensor:  # 原作者，这里得到了context之后一个linear得到了时间间隔分布的参数，用另外一个linear得到了是哪一个事件，相当于就是独立建模
        """Compute log-likelihood for a batch of sequences.#所以，作者在这里当然改了，改成了时间依赖于事件类型的结果。

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """

        features = self.get_features(batch)  # [bsize,seqlen+1,hdim]
        context = self.get_context(
            features)  # [bsize,seqlen+1-1+1,hdim]为什么少了一个，因为将最后一个删除了，为什么多了一个，因为有了context_init相当于要预测第一个事件。
        masks = batch.masks - 1
        brange=list(range(len(masks)))
        trunc = True
        if trunc:
            # 我们只需要上面的最后一个即可。其实有一个rnn被丢弃了，上面有,remove。
            context = context[brange,masks].unsqueeze(1)  # [bsize,1,hdim]
        inter_time_dist = self.get_inter_time_dist(context)  # 这个其实并不关键好像，无非就是得到分布的参数，然后构建那个lognormal分布。
        # inter_time_dist=inter_time_dist.to(features.device)
        inter_times = batch.inter_times.clamp(1e-10)  # 这个默认就是最小值，然后这个间隔时间别忘了，还没有log的。本文的get_features那里log，然后减去均值。
        if trunc:  #
            # 上面的那个形状应该是[bsize,seqlen]，同理，我们也是只要最后一个。
            inter_times = inter_times[brange, masks].unsqueeze(1)
        marks=batch.marks
        if trunc:
            marks=marks[brange,masks].unsqueeze(1)#[bsize,1]
        multi_marks = one_hot(marks,
                              num_classes=self.num_marks).float()  # (batch_size, seq_len, num_marks)总之就也是增加一维的意思。那个class我猜其实是可以不写的。

        inter_times.unsqueeze_(-1)  # (batch_size, seq_len, 1)
        inter_times = inter_times.expand(inter_times.shape[0], inter_times.shape[1],
                                         self.num_marks).clone()  # (batch_size, seq_len, num_marks)
        # 上面这个严格来说，inter以及marks其实都是【bsize,seqlen+1]，末尾多了一个T，至于那个mark就是填充了0而已。
        # print(inter_times.device)
        log_p = inter_time_dist.log_prob(
            inter_times)  # (batch_size, seq_len, num_marks)然后，上面的expand和repeat差不多，repeat是倍数，而expand是需要多少维，从不同角度描述罢了。
        # log_p = log_p.to(features.device)#本来不会先cpu然后又todevice的，但是那个分布好像不支持gpu啊。  # (batch_size, seq_len, num_marks)然后，上面的expand和repeat差不多，repeat是倍数，而expand是需要多少维，从不同角度描述罢了。

        # 其实还是那个点，这里是要得到p(t|m=1,h)的值，所以上面要复制expand，num_marks个。上面这个直接就动手计算了。log_prob是内置的，本身就是 (batch_size, seq_len, num_marks)个分布，所以带入这么多个inter_time也很正常。
        pos_log_p = log_p * multi_marks  # (batch_size, seq_len, num_marks)#这个其实就等价于索引啊，这个log_p是得到了所有事件类型下当前时间的概率，即p(t|m=k,h)，然后这里乘以了一个one-hot，就相当于取出了那个正确的k,当然是希望这个东西越大越好。其实有一个点，如果只是希望这个越大越好的话，那么基于其他事件类型的分布，就没有利用上，这个作者相当于是白做了，说真的。
        neg_log_p = log_p * (1 - multi_marks)  # 这个作者没有使用，不用管，你要说，这个东西代表了什么，代表了基于其他事件类型的logp。
        #
        # Survival probability of the last interval (from t_N to t_end).
        # You can comment this section of the code out if you don't want to implement the log_survival_function
        # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        # but the difference shouldn't be significant if you are working with long sequences.
        # 上面那个生存概率也是重写的，在那个distrbutions下面有，也是按照标准流公式求解，F(y)，但是我没有搞懂，生存其实只有最后一个时间点需要，这里带入全部干嘛？可以看上面，取出来了最后一个，但是只是吐槽，浪费时间。
        # 这里还是要反复梯形，intertime是包括最后一个T的，但是mask不包括。它上面，计算生存函数尽然这么搞，有没有搞错啊我去。他这里的生存概率很扯淡，是p(t>tend|m=mend,h)本来由于两个事件类型，会有两个生存，它上面就是直接选用了最后一个mark（不是填充的），我觉得很有问题啊。因为那个h其实就包括了mend的信息，它相当于只是索引了，选了其中一个作为生存概率。
        mark_logits = torch.log_softmax(self.mark_linear(context),
                                        dim=-1)  # (batch_size, seq_len, num_marks)现在才来这个，不过其实确实和我那个不一样，我那个是知道m=k的情况下，他这里是建模分布，我这里是建模某一个instance的具体概率值。
        tot_nll = (pos_log_p + mark_logits * multi_marks).sum(-1)  # (batch_size, seq_len)
        tot_nll = tot_nll.sum(-1)  # (batch_size)
        # 上面第二哥这个我像就不用管了，后面有那个mask。然后那个加法其实就是联合分布了。然后上面这个就是加上最后一部分。
        mark_dist = Categorical(logits=mark_logits)  # 我去，竟然还有这个，这个是用来干嘛的。暂时没有看懂，上面不是计算了mark_logits嘛？还要分类干嘛呢？
        log_mark = mark_dist.log_prob(
            marks)  # 又是一脸懵逼，[bsize,seqlen]取出来的应该就是概率，和mark_logits的差别也就在于上面那个*multi_marks了吧。实在看不懂。

        mark_class_joint = ((log_p + mark_logits).argmax(
            -1) ).float()  # (batch_size, seq_len)#这个其实是预测了下一个事件是什么，但是有一个问题，其基于了下一次发生事件的时间。从而有了logp，然后argmax，所以，它这个其实才是真正的选择了哪个事件，只是可惜，我们的损失函数并没有分类这一项。
        log_time = pos_log_p.sum(-1)  # [bsize,seqlen,2].sum()其实就是得到p(t|m=k,h)这个k是正确的k。然后mask就是取出需要的。
        # 那么问题来了，它到底在干什么，其实就是在得到log_time,log_mark，想要有分别的指标而已，到时候更好评价。
        return tot_nll, log_time, log_mark, mark_class_joint
    
    def sample(self, t_end: float, batch_size: int = 1, context_init: torch.Tensor = None) -> dpp.data.Batch:
        """Generate a batch of sequence from the model.

        Args:
            t_end: Size of the interval on which to simulate the TPP.
            batch_size: Number of independent sequences to simulate.
            context_init: Context vector for the first event.
                Can be used to condition the generator on past events,
                shape (context_size,)

        Returns;
            batch: Batch of sampled sequences. See dpp.data.batch.Batch.
        """
        if context_init is None:
            # Use the default context vector
            context_init = self.context_init#一般来说，我们当然都是这个啦。
        else:
            # Use the provided context vector
            context_init = context_init.view(self.context_size)
        next_context = context_init[None, None, :].expand(batch_size, 1, -1)#每一个序列一开始都是这个初始context，复制了一遍.[bsize,1,hdim]
        inter_times = torch.empty(batch_size, 0)#t_end是100，这个也是一样的。这个shape是0很有意思哈。
        marks = torch.empty(batch_size, 0, dtype=torch.long)#又是这个。

        generated = False
        while not generated:#这个玩意一模一样啊。
            inter_time_dist = self.get_inter_time_dist(next_context) # (batch_size, 1, num_marks)   还真是没有毛病，p(t|m=1,h)p(t|m=2,h)都会有，所以有这么多个分布。
            next_inter_times = inter_time_dist.sample()  # (batch_size, 1, num_marks)   sample，直接sample正态分布，然后变换就行了，得到了时间，也没有毛病。
            #上面形状也的确没有问题。
            mark_logits = torch.log_softmax(self.mark_linear(next_context), dim=-1)  # (batch_size, 1, num_marks)
            mark_dist = Categorical(logits=mark_logits)#上面mark_linear之后是[bsize,1,2]softmax保持不变，这里构造分布，然后采样，我还以为是采用最大的方式。
            next_marks = mark_dist.sample()  # (batch_size, 1)之后一个了，也没有毛病，这样的话，选出了mark，上面的next_times其实还需要进行赛选一下，根据这个mark，去除掉某一个时间。
            marks = torch.cat([marks, next_marks], dim=1)#没毛病，存储起来。
            #我唯一的一个疑问，为什么上面的next_inter_times不会是log版本的呢？当然不会是，因为我们获取密度的时候用的是没有log的，只有输入rnn是log了，需要搞清楚啊。
            next_inter_time = torch.gather(next_inter_times.squeeze(), dim=-1, index=next_marks) #果然，这里就是要取出来了。[bsize]
            inter_times = torch.cat([inter_times, next_inter_time], dim=-1)  # (batch_size, seq_len)又是存在一起。

            with torch.no_grad():#这个应该是多余的，采样就只发生在eval的时候，这个函数外层早就nograd了。
                generated = inter_times.sum(-1).min() >= t_end#没毛病，时间间隔相加看一下是不是比t_end更大了。一个点，这里写的是min，也就是说要所有的序列都超过的时候才算完成整个bsize的采样。

            batch = Batch(inter_times=inter_times, mask=torch.ones_like(inter_times), marks=marks)#我去，这里又在构造新的batch。搞不懂，本函数最开始都没有构造，这里为什么要构造呢？
            features = self.get_features(batch)  # (batch_size, seq_len, num_features)#我明白了为什么要构造，因为它这里的rnn不是我那种cell，所以必须每一次都要重新传播，不能接着上一次的隐向量继续干。
            context = self.get_context(features, remove_last=False)  # (batch_size, seq_len, context_size)
            next_context = context[:, [-1], :]  # (batch_size, 1, context_size)这个也没有毛病，其实其只有这个有用。
        #我没想到，上面这个for循环尽然还要很久呢。我才明白，是因为bsize是2万多，太大了。
        arrival_times = inter_times.cumsum(-1)  # (batch_size, seq_len)这个不用看，当然就是得到标准时间，原来是[1,2,2]现在是[1,3,5]
        inter_times = diff(arrival_times.clamp(max=t_end), dim=-1)#这里其实就是删除最后面那个大于t的，好像每一个序列都一定会大于t_end。对的，但是大于的个数不一样，有的提早结束了，但是由于保持bsize，还要继续往前冲。
        mask = (arrival_times <= t_end).float()  # (batch_size, seq_len)上面这个就完全是之前训练得时候的标准输入了。后面有大量的填充0，而且一般至少有一个。这是因为那个tend的关系。
        if self.num_marks > 1:#上面这个是得到mask，在t_end以及之前的都是真实事件，我都忘了训练时，最后一个是否有mask了。最后一个没有mask，其不是事件。
            marks = marks * mask  # (batch_size, seq_len)#倒也没有毛病哈。
        return Batch(inter_times=inter_times, mask=mask, marks=marks)#最终组织。