import torch
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt


class MnistSubset():
    def __init__(self, selnum=[2000]*10, mode="train"):
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.data = []
        self.labels = []
        self.selnum = selnum
        classes = [0,1,2,3,4,5,6,7,8,9]

        if mode == "train":
            self.transform = transform_train
            self.mnist = datasets.MNIST(root='F:\代码\ljh\data', download=True, train=True)
            data_source = self.mnist.train_data
            label_source = self.mnist.train_labels

            for c in classes:
                tmp_idx = np.where(label_source == c)[0]  #找出相应类的索引
                tmp_idx = torch.from_numpy(tmp_idx) # 转化为tensor
                seln = selnum[c]
                img = data_source[tmp_idx[:seln]]
                label = label_source[tmp_idx[:seln]]
                self.data.append(img)
                self.labels.append(label)
            self.data = torch.cat(self.data, dim=0)  # Tensor, [xx,28,28]
            self.labels = torch.cat(self.labels, dim=0)  # Tensor, [xx]

        else:
            self.transform = transform_test
            self.mnist = datasets.MNIST(root='mnistData', train=False, download=False)
            self.data = self.mnist.test_data
            self.labels = self.mnist.test_labels

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy())  # model L表示灰度图像, 对于0-255的灰度图 model='L'没有影响
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def getStatic(self):
        labels = self.labels.numpy()
        class_labels, counts = np.unique(labels, return_counts=True)  #自动会排序 ( return_counts = True 返回唯一数组在原数组出现次数)
        return class_labels, counts

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(10):
            cls_num_list.append(self.selnum[i])
        return cls_num_list

def showStatic(class_labels, counts):
    plt.bar(class_labels, counts)
    plt.xlabel('Labels', fontsize=11)
    plt.ylabel('Number of samples', fontsize=11)
    plt.xticks(class_labels, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
    #plt.savefig('paperData/cifar10/实验结果svg/Data-model2-0.4.png')
    #plt.savefig('paperData/cifar10/实验结果svg/Data-model2-0.4.svg')
    plt.show()

if __name__ == '__main__':
    # 由于用到随机数，固定随机数测试
    seed = 0
    random.seed(seed)
    np.random.seed(seed) #固定数组
    torch.cuda.manual_seed(seed) #GPU设置随机数

    selnumber = np.linspace(6000, 60, 10, dtype=int)
    #selnumber = [2000]*10
    trainset = MnistSubset(selnum=selnumber, mode='train')
    #A = trainset[0]
    trainloader = iter(trainset)
    data,label = next(trainloader)
    print('Image shape:', data.shape)
    class_labels, counts = trainset.getStatic()
    showStatic(class_labels, counts )

    cls_num_list = trainset.get_cls_num_list()
    print('class_num_list:', cls_num_list)
    print(len(trainset))
