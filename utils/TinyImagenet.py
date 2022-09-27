import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import os, numpy as np
# from TinyNetPretrain import train_model
from torch.utils.data import Dataset, DataLoader


# Split Training dataset into training (size: 450) and validation (size: 50).
# Labels in the original validation set are all zeros.
def split_train_valid(data_dir='/media/omnisky/HDisk5/dataset/tiny-imagenet-200', n_sample=50):
    data_transforms = transforms.Lambda(lambda image: (np.array(image).astype('uint8')).transpose(2, 0, 1))
    
    image_dataset_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
    
    train_loader = DataLoader(image_dataset_train, batch_size=128, num_workers=4, shuffle=True, drop_last=False)
    
    # for i in range(200):
    #     train_loader.dataset.targets
    
    label_ = []
    x_ = []
    for x, label in train_loader:
        label_.append(label)
        x_.append(x)
    
    label_ = torch.cat(label_)
    x_ = torch.cat(x_)
    label_list = torch.unique(label_).tolist()
    
    torch.manual_seed(0)
    valid_idx = []
    
    for label in label_list:
        label_idx = np.where(label_ == label)[0]
        sub_label_idx = np.random.choice(label_idx, n_sample, replace=False)
        valid_idx.extend(list(sub_label_idx))
    
    valid_x = x_[valid_idx]
    valid_label = label_[valid_idx]
    
    torch.save(valid_x, data_dir + '/valid_x.pt')
    torch.save(valid_label, data_dir + '/valid_label.pt')
    
    train_idx = list(set(range(len(label_))) - set(valid_idx))
    train_x = x_[train_idx]
    train_label = label_[train_idx]
    
    torch.save(train_x, data_dir + '/train_x.pt')
    torch.save(train_label, data_dir + '/train_label.pt')


if __name__ == '__main__':
    split_train_valid(data_dir='/media/omnisky/HDisk5/dataset/tiny-imagenet-200', n_sample=50)