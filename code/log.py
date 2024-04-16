import logging
import sys
import os
from datetime import datetime
def get_logger(args):
    logger=logging.getLogger("logger")
    #上面是取了一个名字，然后我们定义一下处理器。
    print=logging.StreamHandler(sys.stdout)#我们弄到标准输出里面去。但是由于我们不会一直守着，所以也要写到文件里面去。
    logdir=os.path.join(args.pro_path,"log")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    fo = open(os.path.join(logdir, "count.file"), "r")
    count = int(fo.read())
    fo.close()
    fo = open(os.path.join(logdir, "count.file"), "w")
    fo.write(str(count + 1))
    fo.close()
    args.count=count
    # logname = os.path.join(logdir, str(count) +"-"+ str(datetime.now()) + ".log")  # 屈服了，这个有一个好处，写文件的时候大家可以各自写各自的，不容易冲突。
    logname = os.path.join(logdir, str(count) +"-"+args.data+ ".log")  # 屈服了，这个有一个好处，写文件的时候大家可以各自写各自的，不容易冲突。
    file=logging.FileHandler(logname)
    #然后我们再分别设置一下日志的格式即可。
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')

    print.setFormatter(formatter)
    file.setFormatter(formatter)
    logger.addHandler(print)
    logger.addHandler(file)#至此，完全搞定了。一个问题就是，是否打印的都要放到文件中呢？是的。一般来说没有什么问题的。
    logger.setLevel(level=logging.INFO)#默认好像不是info，我们这里全局设置一下，那么所有的handler都会是info。
    return logger

