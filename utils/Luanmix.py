from cmath import exp
from re import L
import numpy as np
import torch, math




def reweightingmix( model, criterion, image, label, K=1, num_class_list=[]):
   
    #混合图片
    l = np.random.beta(1,1)
    idx = torch.randperm(image.size(0))#返回从0到n-1的随机整数排列
    image_a, image_b = image, image[idx] #随机抽取样本
    label_a, label_b = label, label[idx] #随机抽取样本标签
    l_list = torch.empty(image.shape[0]).fill_(l).float().cuda()
    mixed_image = l * image_a + (1-l) * image_b
    mixed_image = mixed_image.cuda()
#    mixed_image = torch.rand_like(image).cuda()
#    nm = torch.max(num_class_list)
#    sum = torch.sum(num_class_list,dim=0)



    # Remix：混合标签
    n_i, n_j = num_class_list[label_a].float(), num_class_list[label_b].float()# i类和j类的样本数
    
    for i in range(image.size(0)):
            if n_i[i] / n_j[i] >= K and l >0.5:
                l_list[i] = n_j[i] / (n_i[i] + n_j[i])
#               l_list[i] = math.exp(nj) / (math.exp(nj) + math.exp(ni))
#               l_list[i] = (n_j[i] / nm)**2
#               l_list[i] = n_j[i] / sum  不行
#               mixed_image[i] = l_list[i] * image_a[i] + (1 - l_list[i]) * image_b[i]
               
            if n_i[i] / n_j[i] < K and l < 0.5 :
                l_list[i] = n_j[i] / (n_i[i] + n_j[i])
          
#               l_list[i] = n_j[i] / (n_i[i] + n_j[i])
#               l_list[i] = (n_j[i] / nm)**2
#               l_list[i] = n_j[i] / sum  不行
#               mixed_image[i] = l_list[i] * image_a[i] + (1 - l_list[i]) * image_b[i]

    output = model(mixed_image)
    mixed_label = l_list * label_a + (1-l_list)*label_b # lambda y 混合label
    loss = l_list * criterion(output, label_a) + (1 - l_list) * criterion(output, label_b)
    loss = loss.mean()

    return loss,output,mixed_label