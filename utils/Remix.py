import numpy as np
import torch, math




def remix( model, criterion, image, label, alpha=1.0, KAPPA=3.0, TAU=0.5,num_class_list=[]):
    r"""
    Reference:
        Chou et al. Remix: Rebalanced Mixup, ECCV 2020 workshop.

    The difference between input mixup and remix is that remix assigns lambdas of mixed labels
    according to the number of images of each class.

    Args:
        tau (float or double): a hyper-parameter
        kappa (float or double): a hyper-parameter
        See Equation (10) in original paper (https://arxiv.org/pdf/2007.03943.pdf) for more details.
    """

    #混合图片
    l = np.random.beta(alpha, alpha) # lambda x
    idx = torch.randperm(image.size(0))#返回从0到n-1的随机整数排列
    image_a, image_b = image, image[idx] #随机抽取样本
    label_a, label_b = label, label[idx] #随机抽取样本标签
    mixed_image = l * image_a + (1 - l) * image_b # 图片按照lambda x mixup
    #mixed_image = torch.tensor(mixed_image)
    mixed_image = mixed_image.cuda()
    #feature = model(mixed_image, feature_flag=True)
    output = model(mixed_image)

    # Remix：混合标签
    l_list = torch.empty(image.shape[0]).fill_(l).float().cuda()
    n_i, n_j = num_class_list[label_a], num_class_list[label_b].float()# i类和j类的样本数

    if l < TAU :
        l_list [n_i / n_j >= KAPPA] = 0


    if 1 - l < TAU:
        l_list[(n_i * KAPPA) / n_j <= 1] = 1


    mixed_label = l_list * label_a + (1-l_list)*label_b # lambda y 混合label
    loss = l_list * criterion(output, label_a) + (1 - l_list) * criterion(output, label_b)
    loss = loss.mean()

    return loss,output,mixed_label