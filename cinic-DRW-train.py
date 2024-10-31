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
from utils import make_logger,fix_random
from dataset import IMBALANCECIFAR10, IMBALANCECIFAR100
from sklearn.metrics import confusion_matrix
from utils import remix,saliencymix,cutmix,mix,IMBALANCECINIC10


def main(data_type='cifar10', loss_type='LDAM', train_rule='none', resultFolder='Results/remix',imb_ratio = 0.1,imb_type='exp',mix_rule='none'):
    # if not os.path.exists(resultFolder):
    #    os.makedirs(resultFolder)
    # logfile = 'train-{}-{}__LossType={}_TrainRule={}.log'.format(time.strftime('%m%d-%H_%M'), data_type, loss_type, train_rule)

    
    # Datatype = 'cifar10'
    # Losstype = 'LDAM'
    # train_rule = 'None'
    seed = 0
    fix_random(seed)

    learning_rate = 0.1
    epochs = 200
    #imb_ratio = 0.1 
    best_acc1 = 0


    
    if not os.path.exists(resultFolder):
        os.makedirs(resultFolder)
    logfile = 'train-{}-{}__LossType={}_TrainRule={}_ImbRatio={}_Epochs={}.log'.format(time.strftime('%m%d-%H_%M'), data_type, loss_type, train_rule, imb_ratio,epochs)
    logger = make_logger(resultFolder, logfile)



    
    logger.info("=> creating model ")
    num_classes = 100 if data_type == 'cifar100' else 10
    use_norm = True if loss_type == 'LDAM' else False
    model = models.ResNet18(num_classes=num_classes, use_norm=use_norm)
    if torch.cuda.is_available():
        model = model.cuda()

    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-4)

    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),  
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835)),
    ])

    if data_type == 'cinic10':
        train_dataset = IMBALANCECINIC10(root='/home/cvk4_n1/douli/cinic-10/train', transform=transform_train,
                                         imb_type=imb_type, imb_factor=imb_ratio)

        val_dataset = datasets.ImageFolder(root='/home/cvk4_n1/douli/cinic-10/valid',
                                                       transform=transform_val)

    else:
        print('NO such dataset!')
        return
    cls_num_list = train_dataset.get_cls_num_list()
    #print('cls num list:')
    #print(cls_num_list)
    logger.info('cls num list:')
    logger.info(cls_num_list)


    #train_sampler = None
    #if train_rule == "Resample":
    #    train_sampler = ImbalancedDatasetSampler(train_dataset)  
    #    per_cls_weights = None



    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr=learning_rate)  


        if train_rule == 'none':
            train_sampler = None
            per_cls_weights = None
        elif train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset) 
            per_cls_weights = None
        elif train_rule == 'Reweight': 
            train_sampler = None
            beta = 0.999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        elif train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0.0999,0.999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        elif train_rule == 'DRS':
            if epoch >= 160:
                train_sampler = ImbalancedDatasetSampler(train_dataset)
                per_cls_weights = None
            else:
                train_sampler = None
                per_cls_weights = None
        else:
            warnings.warn('Sample rule is not listed')
      
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



        #train(train_loader, model, criterion, optimizer, epoch,logger)

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # train
        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
           
            data_time.update(time.time() - end)

            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            #  rebalance mixup
            if mix_rule == 'none':
                output = model(input)
                loss = criterion(output, target)
            elif mix_rule == 'mix':
                loss, output, target = mix(model=model, criterion=criterion, image=input, label=target,
                                           num_class_list=torch.tensor(cls_num_list))
            elif mix_rule == 'remix':
                loss, output, target = remix(model=model, criterion=criterion, image=input, label=target,
                                           num_class_list=torch.tensor(cls_num_list))
            elif mix_rule == 'cutmix':
                loss, output, target = cutmix(model=model, criterion=criterion, image=input, label=target,
                                             num_class_list=torch.tensor(cls_num_list))
            elif mix_rule == 'ours':
                loss, output, target = saliencymix(model=model, criterion=criterion, image=input, label=target,
                                              num_class_list=torch.tensor(cls_num_list))


            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

       
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

       
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1,
                    lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
                logger.info(output)





        # test--------
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

                output = model(input)  # output
                loss = criterion(output, target)

                
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

              
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(output, 1)
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
            cf = confusion_matrix(all_targets, all_preds).astype(float)  
            cls_cnt = cf.sum(axis=1)  
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                      .format(flag=flag, top1=top1, top5=top5, loss=losses))
            out_cls_acc = '%s Class Accuracy: %s' % (
                flag, (np.array2string(cls_acc, separator=',',
                                       formatter={'float_kind': lambda x: "%.3f" % x})))
            logger.info(output)
            logger.info(out_cls_acc)

    
        is_best = top1.avg > best_acc1
        best_acc1 = max(top1.avg, best_acc1)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1) 
        # print(output_best)
        logger.info(output_best)

def adjust_learning_rate(optimizer, epoch, lr=0.1): 
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 180:
        lr = lr * 0.0001
    elif epoch > 160:
        lr = lr * 0.01
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    #main(data_type='cinic10', loss_type='CE', train_rule='none',resultFolder='Results',imb_ratio = 0.1)
    Fire(main)
