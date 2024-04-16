import tick
from tick.hawkes import SimuHawkes, HawkesKernelExp
from tqdm.auto import tqdm
import numpy as np
import torch


def generate_points(mu, alpha, decay, window, seed, dt=0.01):
    """
    Generates points of an marked Hawkes processes using the tick library
    """

    n_processes = len(mu)#这个就是说有多少个事件类型相互影响，作者这里都是两个。
    hawkes = SimuHawkes(n_nodes=n_processes, end_time=window, verbose=False, seed=seed)#我勒个去，这里应该是构造一个对象。恐怕得学习一下api了。
    #两个过程，也给定了一个终止时间100。verbose就是打印应该，不用管，还有种子，否则两次模拟，一定会是一样的。
    for i in range(n_processes):#第i个事件类型。
        for j in range(n_processes):#二重循环，没有毛病，这么多个使劲啊，相互影响。
            hawkes.set_kernel(i=i, j=j, kernel=HawkesKernelExp(intensity=alpha[i][j] / decay[i][j], decay=decay[i][j]))
        hawkes.set_baseline(i, mu[i])#注意到，上面指定了那个核是指数核，也就是alpha*beta*exp(-beta(t-tj))，但是上面感觉是不是写错了，不需要除以decay好像，直接alpha就行了。
        #上面是指定了第i个事件类型的base rate,然后上面是指定了第i个事件类型和其他的核的情况。
    hawkes.track_intensity(dt)#这个老实说，不知道是干嘛用的。dt=0.01，这个又是什么鬼，这种模拟感觉是有算法的，我没有看过，怕是不好讲啊。
    hawkes.simulate()#上面莫非是离散化各个事件类型的强度？间隔就是上述dt，我猜应该是，从而每次都可以计算强度，然后判断是否要生成这个事件类型。具体细节以后再说吧。
    return hawkes.timestamps#这个是只返回事件戳嘛？那么事件类型呢？也返回了其实。其是一个两个元素的列表，第一个元素是一个列表，存储了第一个事件类型的发生时间，第二个同理。


def hawkes_helper(mu, alpha, decay, window, in_seed, in_range):
    times_marked = [generate_points(mu=mu, alpha=alpha, decay=decay, window=window, seed=in_seed + i) for i in
                    tqdm(range(in_range))]#产生多少条序列，所以说，generate这个函数是用来产生一条序列的，然后seed每次都会变，可以的。
    records = [hawkes_seq_to_record(r) for r in (times_marked)]#这个是什么鬼，可能是组织成一个比较统一的数据结构。
    return records


def hawkes_seq_to_record(seq):
    times = np.concatenate(seq)#这个直接将两个序列合并，肯定是要排序等下。
    labels = np.concatenate([[i] * len(x) for i, x in enumerate(seq)])#这个也没有毛病，和seq的数据结构一模一样，但是一个存储事件类型，0，1，一个是存储时间。
    sort_idx = np.argsort(times)#这个应该是从小到大排列的。
    times = times[sort_idx]#相当于就是排了顺序。
    labels = labels[sort_idx]#牛批，没毛病，把事件类型也对齐了。
    record = [
        {"time": float(t),
         "labels": (int(l),)} for t, l in zip(times, labels)]#没有毛病，每一个事件都组织成了字典，其实直接m,t元组就好了啊，他这里太客气了。
    return record


def combine_splits(d_train, d_val, d_test):
    sequences = []

    for dataset in ([d_train, d_val, d_test]):#这里相当于是一个一个来的。
        for i in range(len(dataset)):#这里是每一个序列。
            event_dict = {}
            arrival_times = []
            marks = []
            for j in range(len(dataset[i])):#这里是序列中的每一个事件。
                curr_time = dataset[i][j]['time']
                curr_mark = dataset[i][j]['labels'][0]
                arrival_times.append(curr_time)
                marks.append(curr_mark)

            event_dict['t_start'] = 0
            event_dict['t_end'] = 100
            event_dict['arrival_times'] = arrival_times
            event_dict['marks'] = marks

            sequences.append(event_dict)#加入这个序列中。最后相当于每一个序列都组织成了上述格式，然后没有了train/val/test，就完全是合并了。这个好像是之前某个人的数据格式，类似于TGAT。

    return sequences


def dataset_helper(mu, alpha, beta, window, seed, train_size, val_size, test_size, save_path):
    train_seed = seed
    val_seed = seed + train_size#卧槽，还搞这招，也是无语了，其实大可不必，纠结这个seed我觉得。
    test_seed = seed + train_size + val_size

    d_train = hawkes_helper(mu, alpha, beta, window, train_seed, train_size)#所以核心代码是在这里，这里是要生成n个序列，然后每一个序列都是[0,T],这样子。
    d_val = hawkes_helper(mu, alpha, beta, window, val_seed, val_size)#另外一个其实也要生成，那就是事件类型，所以其会是[(m1,t1),...,]的形式。
    d_test = hawkes_helper(mu, alpha, beta, window, test_seed, test_size)

    sequences = combine_splits(d_train, d_val, d_test)#这个不会是重新分配上述序列把？按理上述已经划分了train/val/test。坑的地方在于，他这里确实合并了，但是并没有再划分。每一条sequence都成为了一个列表元素。
    dataset = {'sequences': sequences, 'num_marks': len(mu)}
    torch.save(dataset, save_path)


### Hawkes Ind.生成的第一个数据集。我算是明白了，原来它这个是多事件霍克斯过程，这个mu是两个元素，就是说，它这个有两种事件类型。
mu = [0.1, 0.05]
alpha = [[0.2, 0.0], [0.0, 0.4]]#然后alpha,beta，alpha*beta*exp(beta*(t-tx)),其中tx可能是事件1，也可能是事件2，此时，alpha,beta就会不一样，表示对当前强度的影响
beta =  [[1.0, 1.0], [1.0, 2.0]]
#那么上面为什么会是alpha是对角阵，就说明，事件1对事件2没有影响，其实如果beta是对角阵，根据上面那个公式也可以得到结论。我好奇的时候，为什么exp里面有beta，外面还要一个。
window = 100#这个就是T的意思。终止时间，一个序列采样到这个时候就结束。
seed = 0
train_size = 14745
val_size = 4915
test_size = 4916
save_path = '../data/synth/hawkes_ind.pkl'

dataset_helper(mu, alpha, beta, window, seed, train_size, val_size, test_size, save_path)

### Hawkes Dep. I #生成的第二个数据集
mu = [0.1, 0.05]#这里dep的意思就是2个事件类型，但是有相互依赖。之前的是相互事件不会产生什么激励。
alpha = [[0.2, 0.1], [0.2, 0.3]]
beta =  [[1.0, 1.0], [1.0, 1.0]]

window = 100#这个就是T的意思。
seed = 0
train_size = 14745
val_size = 4915
test_size = 4917
save_path = '../data/synth/hawkes_dep_I.pkl'

dataset_helper(mu, alpha, beta, window, seed, train_size, val_size, test_size, save_path)
### Hawkes Dep. II 生成的第三个数据集，也是依赖，但是下面这个已经变成了5个事件类型了。
mu = [0.713, 0.057, 0.844, 0.254, 0.344]

alpha = [[0.689, 0.549, 0.066, 0.819, 0.007],
         [0.630, 0.000, 0.457, 0.622, 0.141],
         [0.134, 0.579, 0.821, 0.527, 0.795],
         [0.199, 0.556, 0.147, 0.030, 0.649],
         [0.353, 0.557, 0.892, 0.638, 0.836]]


beta = [[9.325, 9.764, 2.581, 4.007, 9.319],
        [5.759, 8.742, 4.741, 7.320, 9.768],
        [2.841, 4.349, 6.920, 5.640, 3.839],
        [6.710, 7.460, 3.685, 4.052, 6.813],
        [2.486, 2.214, 8.718, 4.594, 2.642]]

window = 100
seed = 0
train_size = 18000
val_size = 6000
test_size = 6000
save_path = '../data/synth/hawkes_dep_II.pkl'

dataset_helper(mu, alpha, beta, window, seed, train_size, val_size, test_size, save_path)
