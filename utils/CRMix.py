import numpy as np
import torch, math
import copy


def CRMix(model, criterion, image, label, alpha=1.0, K=3.0, TAU=0.5, num_class_list=[]):
    l = np.random.beta(alpha, alpha)  # lambda
    idx = torch.randperm(image.size(0))  # 返回从0到n-1的随机整数排列
    image_a, image_b = image, image[idx]  # 随机抽取样本
    label_a, label_b = label, label[idx]  # 随机抽取样本标签
    image_a, image_b = image_a.cuda(), image_b.cuda()
    label_a, label_b = label_a.cuda(), label_b.cuda()
    
    n_i, n_j = num_class_list[label_a], num_class_list[label_b].float()
    aax1, aay1, aax2, aay2 = cut_rand_bbox(image.size(), l)
    lam = 1 - ((aax2 - aax1) * (aay2 - aay1) / (image.size()[-1] * image.size()[-2]))
    l_label = torch.empty(image_a.shape[0]).fill_(lam).float().cuda()
    
    #    image[:, :, aax1:aax2, aay1:aay2] = image_b[:, :, aax1:aax2, aay1:aay2]
    if lam < TAU:
        l_label[n_i / n_j >= K] = 0
        image[n_i / n_j >= K] = image_b[n_i / n_j >= K]
        image[n_i / n_j < K, :, aax1:aax2, aay1:aay2] = image_b[n_i / n_j < K, :, aax1:aax2, aay1:aay2]
    if lam > TAU:
        l_label[(n_i * K) / n_j <= 1] = 1
        image[(n_i * K) / n_j <= 1] = image_a[(n_i * K) / n_j <= 1]
        image[(n_i * K) / n_j > 1, :, aax1:aax2, aay1:aay2] = image_b[(n_i * K) / n_j > 1, :, aax1:aax2, aay1:aay2]
    
    mixed_image = image
    mixed_image = mixed_image.cuda()
    output = model(mixed_image)
    loss = criterion(output, label_a) * l_label + criterion(output, label_b) * (1 - l_label)
    loss = loss.mean()
    mixed_label = l_label * label_a + (1 - l_label) * label_b
    
    return loss, output, mixed_label


def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    
    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]
    
    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def cut_rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform

    # cx = np.random.randint(W)
    # cy = np.random.randint(H)
    cx = np.random.randint(low=cut_w // 2, high=W - cut_w // 2)
    cy = np.random.randint(low=cut_h // 2, high=H - cut_h // 2)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2