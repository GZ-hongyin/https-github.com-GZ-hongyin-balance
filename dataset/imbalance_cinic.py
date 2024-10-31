import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



class IMBALANCECINIC10(torchvision.datasets.ImageFolder):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0,
                 transform=None, target_transform=None):
        super(IMBALANCECINIC10, self).__init__(root, transform, target_transform)
        self.data = np.array(self.samples)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        #print(new_data)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def getStatic(self):
        #labels = self.targets.numpy()
        labels = np.array(self.targets)
        class_labels, counts = np.unique(labels, return_counts=True)
        return class_labels,counts

    def __len__(self):
        """
        Original Code is self.samples, we change to self.data
        Thus override this method
        """
        return len(self.data)


def showStatic(class_labels, counts):
    plt.bar(class_labels, counts)
    plt.xlabel('Labels', fontsize = 11)
    plt.ylabel('Number of Samples', fontsize =11)
    plt.xticks(class_labels,('0','1','2','3','4','5','6','7','8','9'))
    plt.show()




if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECINIC10(root='/home/cvk4_n1/douli/cinic-10/'+'train', transform=transform,imb_factor=0.01)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    cls_num_list = trainset.get_cls_num_list()
    print('Image shape:', data.shape)
    print('class_num_list:', cls_num_list)
    print(len(trainset))

    class_labels,counts = trainset.getStatic()
    showStatic(class_labels,counts)
    testset = torchvision.datasets.ImageFolder(root='/home/cvk4_n1/douli/cinic-10/'+'test')
    print(len(testset))
