from pathlib import Path
import warnings
import time
import torch.nn as nn
import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.utils.data
import numpy as np
import os
from fire import Fire
from losses import FocalLoss, LDAMLoss
from utils import ImbalancedDatasetSampler, AverageMeter, accuracy
from utils import make_logger, fix_random
from dataset import IMBALANCECIFAR10, IMBALANCECIFAR100
from sklearn.metrics import confusion_matrix
from utils import mix, saliencymix, remix, cutmix, CRMix
import matplotlib.pyplot as plt
from utils import get_combo_loader


def main(data_type='cifar10', loss_type='CE', train_rule='none', resultFolder='Results',
         imb_ratio=0.01, imb_type='exp', mix_rule='none',alpha = 0.3):   
#    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    seed = 0
    fix_random(seed)
    
    learning_rate = 0.1
    epochs = 300
    # imb_ratio = 0.01  # 不平衡率100：0.01
    best_acc1 = 0
    
    # 日志
    if not os.path.exists(resultFolder):
        os.makedirs(resultFolder)
    logfile = 'train-{}-{}__LossType={}_TrainRule={}_ImbRatio={}_Epochs={}_Alpha={}.log'.format(time.strftime('%m%d-%H_%M'),
                                                                                       data_type, loss_type, train_rule,
                                                                                       imb_ratio, epochs, alpha)
    logger = make_logger(resultFolder, logfile)
    
    # 构建网络
    logger.info("=> creating model ")
    num_classes = 100 if data_type == 'cifar100' else 10
    use_norm = True if loss_type == 'LDAM' else False
    model = models.ResNet34(num_classes=num_classes, use_norm=use_norm)
    if torch.cuda.is_available():
        #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()  # all GPU
        model = model.cuda()
    
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-4)
    
    # 数据集加载
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 三个通道
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if data_type == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=imb_type, imb_factor=imb_ratio,
                                         rand_number=0, train=True, download=True,
                                         transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    
    elif data_type == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=imb_type, imb_factor=imb_ratio,
                                          rand_number=0, train=True, download=True,
                                          transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    
    else:
        print('NO such dataset!')
        return
    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    logger.info('cls num list:')
    logger.info(cls_num_list)
    
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr=learning_rate)  # 调整学习率
        
        # 调整训练策略
        if train_rule == 'none':
            train_sampler = None
            per_cls_weights = None
        elif train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)  # 重采样
            per_cls_weights = None
        elif train_rule == 'Reweight':  # 有效样本重加权
            train_sampler = None
            beta = 0.999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        elif train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 225
            betas = [0.0999, 0.999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        else:
            warnings.warn('Sample rule is not listed')
        
        # 设置损失函数
        if loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda()
        elif loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda()
        elif loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda()
        else:
            warnings.warn('Loss Type is not listed')
            return
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=(train_sampler is None),
            num_workers=8, pin_memory=True, sampler=train_sampler)
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=100, shuffle=True,
            num_workers=8, pin_memory=True)

        combo_loader = get_combo_loader(train_loader)
        
        
        
        # -----------训练------------
        # train(train_loader, model, criterion, optimizer, epoch)
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        
        # train
        model.train()
        
        end = time.time()  # 当前时间
        for i, batch in enumerate(combo_loader):
            # 测量数据加载的时间
            data_time.update(time.time() - end)
            
            #lam = (epoch/epochs)
            #lam = (epoch/epochs)**2
            lam = np.random.beta(a=alpha,b=1)
            inputs, labels = batch[0][0], batch[0][1]
            balanced_inputs , balanced_labels = batch[1][0],batch[1][1]
            
            if torch.cuda.is_available():
                inputs,labels = inputs.cuda(non_blocking=True) ,labels.cuda(non_blocking=True)
                balanced_inputs , balanced_labels = balanced_inputs.cuda(non_blocking=True), balanced_labels.cuda(non_blocking=True)
                
                inputs = (1-lam) * inputs + lam * balanced_inputs
                mixed_labels = (1-lam) * labels + lam * balanced_labels
                mixed_labels = mixed_labels.long()
                
#                del balanced_inputs
#                del balanced_labels
                
                output = model(inputs)
                loss = (1-lam) * criterion(output,labels) + lam * criterion(output,balanced_labels)
                loss = loss.mean()
            
            # mixup
#            if mix_rule == 'none':
#                output = model(input)
#                loss = criterion(output, target)
#            elif mix_rule == 'mix':
#                loss, output, target = mix(model=model, criterion=criterion, image=input, label=target,
#                                           num_class_list=torch.tensor(cls_num_list))
#            elif mix_rule == 'remix':
#                loss, output, target = remix(model=model, criterion=criterion, image=input, label=target,
#                                             num_class_list=torch.tensor(cls_num_list))
#            elif mix_rule == 'cutmix':
#                loss, output, target = cutmix(model=model, criterion=criterion, image=input, label=target,
#                                              num_class_list=torch.tensor(cls_num_list))
#            elif mix_rule == 'ours':
#                loss, output, target = CRMix(model=model, criterion=criterion, image=input, label=target,
#                                             num_class_list=torch.tensor(cls_num_list))
            
            # 测量准确率并记录损失
            acc1, acc5 = accuracy(output, mixed_labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            
            # 计算梯度反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 测量时间
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 10 == 0:  # 打印频率
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1,
                    lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
                logger.info(output)
        
        # ---------------测试--------------------
        # acc1 = validate(val_loader, model, criterion, epoch)
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        flag = 'val'
        
        # test
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                if torch.cuda.is_available():
                    input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                
                output = model(input)  # output：各个类的预测概率
                loss = criterion(output, target)
                
                # 准确率记录损失
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))
                
                # 测量时间
                batch_time.update(time.time() - end)
                end = time.time()
                
                _, pred = torch.max(output, 1)  # pred：最大预测概率的索引（类）  _:预测概率
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                if i % 10 == 0:
                    output = ('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))
                    logger.info(output)
            cf = confusion_matrix(all_targets, all_preds).astype(int)  # 混淆矩阵，以矩阵形式将数据集中的记录按照真实的类别与分类模型作出的分类判断两个标准进行汇总
            cls_cnt = cf.sum(axis=1)  # 矩阵每一行的 元素相加
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                      .format(flag=flag, top1=top1, top5=top5, loss=losses))
            out_cls_acc = '%s Class Accuracy: %s' % (
                flag, (np.array2string(cls_acc, separator=',',
                                       formatter={'float_kind': lambda x: "%.3f" % x})))  # 返回一个数组的字符串形式
            logger.info(output)
            logger.info(out_cls_acc)
        
        # 记录最高的acc1
        is_best = top1.avg > best_acc1
        best_acc1 = max(top1.avg, best_acc1)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)  # top1 准确率
        # print(output_best)
        logger.info(output_best)
    
#    torch.save(model.state_dict(),'/media/omnisky/HDisk4/ljh3/ljh/save_model/SAVE.pth')
    

def adjust_learning_rate(optimizer, epoch, lr=0.1):  # 学习率衰减
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 225:
        lr = lr * 0.0001
    elif epoch > 150:
        lr = lr * 0.01
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
#     main(data_type='cifar10', loss_type='CE', train_rule='none',resultFolder='Results',imb_ratio = 0.01,mix_rule = 'ours', imb_type='exp')
    Fire(main)