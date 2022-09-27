import numpy as np
import torch, math




def mix( model, criterion, image, label, alpha=1.0, KAPPA=3.0, TAU=0.5,num_class_list=[]):

    #混合图片
    lx = np.random.beta(alpha, alpha)

    #ly = torch.empty(image.shape[0]).fill_(lx).float().cuda()
    idx = torch.randperm(image.size(0))  # 返回从0到n-1的随机整数排列
    image_a, image_b = image, image[idx]  # 随机抽取样本
    label_a, label_b = label, label[idx]
    image_a, image_b = image_a.cuda(), image_b.cuda()
    label_a, label_b = label_a.cuda(), label_b.cuda()
    #n_i, n_j = num_class_list[label_a], num_class_list[label_b].float()  # i类和j类的样本数
    mixed_image = lx * image_a + (1 - lx) * image_b

    mixed_image = mixed_image.cuda()
    output = model(mixed_image)
    mixed_label = lx * label_a + (1 - lx) * label_b
    loss = lx * criterion(output, label_a) + (1 - lx) * criterion(output, label_b)
    loss = loss.mean()

    return loss, output, mixed_label