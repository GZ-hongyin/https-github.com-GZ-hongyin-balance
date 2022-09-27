import logging
import os
import sys

# 封装标注的logging库：输出到文件和自带显示，自带格式等时间信息, 每次运行都重新开始
def make_logger(save_dir, save_filename, name='tmpExp'): #保存为txt文件
    logger = logging.getLogger(name) #初始化
    logger.setLevel(logging.INFO)#设置日志等级

    ch = logging.StreamHandler(stream=sys.stdout) #建立一个streamhandler来把日志打在CMD窗口
    ch.setLevel(logging.INFO)
    #formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s") #设置日志格式(打印时间，日志级别，日志信息)
    ch.setFormatter(formatter)
    logger.addHandler(ch)#将相应的handler添加在logger对象中

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, save_filename), mode='w') #建立一个filehandler来把日志记录在文件里
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# 简单的logger: 输出到文件和显示, 缺点：如果文件已经存在，新的log会追加在里面，以前的不会删除.
class AddLogger:
    def __init__(self, save_dir, save_filename):
        self.logger_path = os.path.join(save_dir, save_filename)

    def __call__(self, input):
        input = str(input)
        with open(self.logger_path, 'a') as f:  #注意是文件的打开方式为追加
            f.writelines(input+'\n')
        print(input)

if __name__ == '__main__':
    # pass
     output_dir = 'result'
     output_path = 'result'
     logger1 = make_logger(output_dir, 'log1.txt', "Reid_Baseline")
    # logger2 = AddLogger(output_dir, 'log2.txt')
    #
     for k in range(5):
         logger1.info("Using {} GPUS".format(k))
    #     logger2("Using {} GPUS".format(k))