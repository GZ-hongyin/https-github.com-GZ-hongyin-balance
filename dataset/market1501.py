# copy from SphereReID
# modifed by YZ 2020/01/15
#


import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from dataset.random_erasing import RandomErasing

class Market1501(Dataset):
    cls_num = 751
    def __init__(self,data_pth, is_train=True, *args, **kwargs):
        super(Market1501, self).__init__(*args, **kwargs) 

        ## parse image names to generate image ids
        imgs = os.listdir(data_pth) 
        imgs = [im for im in imgs if osp.splitext(im)[-1] == '.jpg']   
        self.is_train = is_train
        self.im_pths = [osp.join(data_pth, im) for im in imgs]
        self.im_infos = {}                                             
        self.person_infos = {}      
        for i, im in enumerate(imgs):
            tokens = im.split('_') 
            im_pth = self.im_pths[i]
            pid = int(tokens[0])
            cam = int(tokens[1][1])
            self.im_infos.update({im_pth: (pid, cam)}) #合并

            if pid in self.person_infos.keys():
                self.person_infos[pid].append(i)
            else:
                self.person_infos[pid] = [i, ]



        self.pid_label_map = {}
        if self.is_train:
          for i, (pid, ids) in enumerate(self.person_infos.items()): 
             self.person_infos[pid] = np.array(ids, dtype = np.int32)
             self.pid_label_map[pid] = i   
        else:
          for i, (pid, ids) in enumerate(self.person_infos.items()): 
            self.person_infos[pid] = np.array(ids, dtype=np.int32)
            self.pid_label_map[pid] = i  

        ## preprocessing
        self.trans_train = transforms.Compose([
                #transforms.Resize((288, 144)),
                #transforms.RandomCrop((256, 128)),
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                #RandomErasing(0.5, mean=[0.0, 0.0, 0.0])
            ])

        ## H-Flip：测试的时候用
        self.trans_no_train_flip = transforms.Compose([
                transforms.Resize((288, 144)),
                transforms.RandomHorizontalFlip(1),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.trans_no_train_noflip = transforms.Compose([
                #transforms.Resize((288, 144)),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        #img_num_list = self.cls_num_list


    def get_img_num_per_cls(self):
        img_num_per_cls = []
        for i, (pid, ids) in enumerate(self.person_infos.items()):
            img_num = len(ids)
            img_num_per_cls.append(img_num)
            img_num_per_cls.sort(reverse=True)
        return img_num_per_cls

    #def gen_imbalanced_data(self, img_num_per_cls):




    def __getitem__(self, idx):
        im_pth = self.im_pths[idx]
        pid = self.im_infos[im_pth][0]
        im = Image.open(im_pth)

        if self.is_train:
            im = self.trans_train(im)
        else:
            im_noflip = self.trans_no_train_noflip(im)
    #        im_flip = self.trans_no_train_flip(im)
            im = im_noflip
            #im = [im_noflip, im_flip]

        #return im, self.pid_label_map[pid], self.im_infos[im_pth] 
        return  im, self.pid_label_map[pid]

    def __len__(self):
        return len(self.im_pths)

    def get_num_classes(self):
        return len(list(self.person_infos.keys()))
          #751

    def get_cls_num_list(self):
        classes = []
        cls_num_list = []
        for i, (pid, ids) in enumerate(self.person_infos.items()):
            img_num = len(ids)
            the_class = pid
            classes.append(the_class)
            cls_num_list.append(img_num)
            cls_num_list.sort(reverse=True)
        return cls_num_list





if __name__ == "__main__":
    ds_train = Market1501('F:\代码\LDAM-DRW-master\datasets\Market-1501-v15.09.15/bounding_box_train')
    ds_test = Market1501('F:\代码\LDAM-DRW-master\datasets\Market-1501-v15.09.15/bounding_box_test', is_train=False)
    #im, lb, pidcam = ds_train[0]
    #im, lb = ds_test[0]
    testloader = iter(ds_test)
    im, lb = next(testloader)
    print('Image shape:', im.shape)
    print('reLabel:', lb)
    #print('pid-cam:', pidcam)



