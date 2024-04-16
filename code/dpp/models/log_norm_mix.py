import torch
import torch.nn as nn

import torch.distributions as D
from dpp.distributions import Normal, MixtureSameFamily, TransformedDistribution
from dpp.utils import clamp_preserve_gradients

from .cdf import RecurrentTPP


class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    We model it in the following way (see Appendix D.2 in the paper):

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
    """
    def __init__(
        self,
        locs: torch.Tensor,
        log_scales: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        transforms = []
        if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)]#可以看到，这里使用了第二次这个玩意了。而且是b,k的形式。
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        transforms.append(D.ExpTransform())
        super().__init__(GMM, transforms)

    @property
    def mean(self) -> torch.Tensor:
        """
        Compute the expected value of the distribution.

        See https://github.com/shchur/ifl-tpp/issues/3#issuecomment-623720667

        Returns:
            mean: Expected value, shape (batch_size, seq_len)
        """
        a = self.std_log_inter_time
        b = self.mean_log_inter_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights + a * loc + b + 0.5 * a**2 * variance).logsumexp(-1).exp()


class LogNormMix(RecurrentTPP):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    The distribution of the inter-event times given the history is modeled with a LogNormal mixture distribution.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        num_mix_components: Number of mixture components in the inter-event time distribution.
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """

#     args,
#     num_marks: int,
#     mean_log_inter_time: float = 0.0,
#     std_log_inter_time: float = 1.0,
#     mean_cum = 0.0,
#     std_cum = 1.0,
#
#
# mean_inter_time: float = 0.0,
# std_inter_time: float = 1.0,
# context_size: int = 32,
# mark_embedding_size: int = 32,
# rnn_type: str = "GRU",
# flownum: int = 30,
# flowlen: int = 2,
    def __init__(#这个初始化函数也改动了。
        self,
            args,
        num_marks: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 32,
        mark_embedding_size: int = 32,
        num_mix_components: int = 16,
        rnn_type: str = "GRU",
            mean_cum=0.0,
            std_cum=1.0,
            mean_inter_time=0.0,
            std_inter_time=1.0,
    ):
        super().__init__(
        args=args,
        # num_marks: int,
        mean_log_inter_time= mean_log_inter_time,
        std_log_inter_time=std_log_inter_time,
            mean_cum=mean_cum,
            std_cum=std_cum,
        mean_inter_time=mean_inter_time,
        std_inter_time=std_inter_time,
            num_marks=num_marks,
            context_size=context_size,
            mark_embedding_size=mark_embedding_size,
            rnn_type=rnn_type,
        )
        self.num_mix_components = num_mix_components
        self.linear = nn.Linear(self.context_size, 3 * self.num_mix_components)#这里改动了，其实就和那个图一样。因为这里是p(t|m,h),也就是说，这里要得到好多个分布p(t|m=1,h),p(t|m=2,h),有的大有的小，然后我们又有p(m=1/2|h)相乘就知道最终选哪个了。

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:#这个作者稍微改了一下，但是大多不变。
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raw_params = self.linear(context)  # (batch_size, m，3 * num_mix_components)
        raw_params = raw_params.reshape(-1,1,self.num_marks,3 * self.num_mix_components)#这里是作者加的，和原文就只有这里有区别好像。
        #[bsize,1,num_marks,3*num_functions]这样没有问题，不过我在好奇，他这里一个context能够包含这么多m=k对应的事件类型的信息嘛？我不相信，所以一定是有改进空间的。但是直接统统一个t（即原作者的独立建模）好像也确实不太好。
        # Slice the tensor to get the parameters of the mixture
        locs = raw_params[..., :self.num_mix_components]#这些都没有毛病，取出均值，方差的log值，以及权重的log值（归一化之后的log）。
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]
        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)#
        log_weights = torch.log_softmax(log_weights, dim=-1)#
        # print(locs.device)
        # print(log_scales.device)
        # print(log_weights.device)
        return LogNormalMixtureDistribution(#混合分布，也没有毛病。
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,#又用了这个东西。其实前面用过一次了，这里又要用一次。
            # std_log_inter_time=self.std_log_inter_time#原来罪魁祸首就在这里，这里应该做的一件事情是将这两个数字给上传到device上去，尽管他们只是一个数字的tensor，也要上传，除非直接数字，不能是一个数字的tensor。
        )
