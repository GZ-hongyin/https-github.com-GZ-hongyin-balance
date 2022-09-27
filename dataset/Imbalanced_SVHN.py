import torchvision
import torchvision.transforms as transforms
import os
import pickle
import scipy.io as sio
import numpy as np


class ImbalanceSVHN(torchvision.datasets.SVHN):
    cls_num = 10

    def __init__(self, root, imb_type='step', imb_factor=0.1, rand_number=0, split='train',
                 transform=None, target_transform=None, download=False):
        super(ImbalanceSVHN, self).__init__(root, split, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # img_max = len(self.data) / cls_num
        img_max = 5000
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
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)
        # shift label 0 to the last (as original SVHN labels)
        # since SVHN itself is long-tailed, label 10 (0 here) may not contain enough images
        # classes = np.concatenate([classes[1:], classes[:1]], axis=0)      #####################
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # print(f"Class {the_class}:\t{len(idx)}")
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.labels = new_targets
        # assert new_data.shape[0] == len(new_targets), 'Length of data & labels do not match!'    #####################
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),  # svhn dataset
        ])
    trainset = ImbalanceSVHN(root='/home/cvk4_n1/douli/ljh/data', split='train', download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    cls_num_list = trainset.get_cls_num_list()
    print(cls_num_list)
    import pdb; pdb.set_trace()
